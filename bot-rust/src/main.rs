// ScrabbleBot bot in Rust, a port of bot-starter/src/index.ts.
//
// First run, with a token from the website:
//   On your team's page, click "Generate token", then run
//   `BOT_NAME=<name> BOT_TOKEN=<token> cargo run --release`.
//
// Or with a nonce from the CLI:
//   1. `spacetime call scrabblebot mint_credential_nonce`
//      `spacetime sql scrabblebot 'SELECT code FROM my_nonces ORDER BY expires_at DESC'`
//   2. `BOT_NAME=<name> BOT_NONCE=<code> cargo run --release`
//
// Either way the bot saves its token to .token-<BOT_NAME>, so later runs
// need only `BOT_NAME=<name> cargo run --release`.
//
// Entering a tournament: `TOURNAMENT_ID=<id>` registers the bot when it
// connects. `THETA=<file>` (default theta.txt here) holds the strategy's
// parameters; it's read at the start of every match, so `cargo run --bin
// tune` can improve a running bot. THETA=theta-search.txt plays the search
// strategy, which also needs market.txt and leave.txt (see below).

mod module_bindings;

use bot_rust::leave::Leave;
use bot_rust::sim::Market;
use bot_rust::strategy::{code, spell, Brain, Rack, Theta, WordIndex};
use module_bindings::*;
use spacetimedb_sdk::{DbContext, Event, Identity, Table, TableWithPrimaryKey, Timestamp};
use std::collections::{HashMap, HashSet};
use std::sync::mpsc;
use std::sync::{Arc, Mutex, MutexGuard, OnceLock};
use std::time::Duration;

fn env(name: &str) -> Option<String> {
    std::env::var(name).ok().filter(|v| !v.is_empty())
}

fn bot_name() -> String {
    env("BOT_NAME").unwrap_or_else(|| "bot".into())
}

fn token_path() -> String {
    format!(".token-{}", bot_name())
}

fn theta_path() -> String {
    env("THETA").unwrap_or_else(|| concat!(env!("CARGO_MANIFEST_DIR"), "/theta.txt").into())
}

// Don't bother bidding on an auction closing sooner than this: the bid
// would arrive late (one-way latency is about 40 ms).
const MIN_TIME_LEFT_MS: i64 = 60;

#[derive(Default)]
struct State {
    my_identity: Option<Identity>,
    my_bot_id: Option<u64>,
    subscribed_as_bot: bool,
    bids: HashMap<u64, i64>, // auction id -> amount we bid
    brains: HashMap<u64, Brain>,
    playing: HashSet<u64>, // matches with a word in flight
}

// Callbacks all run on the connection's one thread, so this lock is never
// contended; it just makes the state reachable from them.
fn state() -> MutexGuard<'static, State> {
    static STATE: OnceLock<Mutex<State>> = OnceLock::new();
    STATE.get_or_init(Default::default).lock().unwrap()
}

macro_rules! log {
    ($($arg:tt)*) => { println!("[{}] {}", bot_name(), format!($($arg)*)) };
}

// MARKET=<file> (default market.txt, written by the tuner) and LEAVE=<file>
// (default leave.txt, from `cargo run --bin learn`) feed the search.
fn market_path() -> String {
    env("MARKET").unwrap_or_else(|| concat!(env!("CARGO_MANIFEST_DIR"), "/market.txt").into())
}

fn leave_path() -> String {
    env("LEAVE").unwrap_or_else(|| concat!(env!("CARGO_MANIFEST_DIR"), "/leave.txt").into())
}

fn data<T>(path: &str, load: fn(&str) -> std::io::Result<T>) -> Option<Arc<T>> {
    match load(path) {
        Ok(v) => Some(Arc::new(v)),
        Err(e) => {
            log!("no {path} ({e})");
            None
        }
    }
}

// A reducer call fails when the module refuses it, often for an expected
// reason: a word played just after its match ended, a bid that missed the
// auction's window. Log it and keep playing.
fn log_failure(what: String) -> impl FnOnce(&ReducerEventContext, Result<Result<(), String>, spacetimedb_sdk::__codegen::InternalError>) + Send + 'static {
    move |_, res| match res {
        Ok(Ok(())) => {}
        Ok(Err(e)) => log!("{what} failed: {e}"),
        Err(e) => log!("{what} failed: {e:?}"),
    }
}

fn resolve_my_bot_id(db: &RemoteTables, st: &State) -> Option<u64> {
    db.bot_credential().identity().find(&st.my_identity?).map(|c| c.bot_id)
}

fn rack_for_match(db: &RemoteTables, match_id: u64) -> Rack {
    let mut rack = [0; 26];
    for h in db.my_rack().iter().filter(|h| h.match_id == match_id) {
        if let Some(&b) = h.letter.as_bytes().first() {
            rack[code(b) as usize] += h.count as u8;
        }
    }
    rack
}

fn balance_in(db: &RemoteTables, st: &State, match_id: u64) -> Option<i64> {
    let me = st.my_bot_id?;
    db.match_participant().iter().find(|p| p.match_id == match_id && p.bot_id == me).map(|p| p.balance)
}

fn is_running(db: &RemoteTables, match_id: u64) -> bool {
    db.match_state().id().find(&match_id).is_some_and(|m| m.status == MatchStatus::Running)
}

fn letters_left(db: &RemoteTables, match_id: u64) -> u32 {
    db.match_state().id().find(&match_id).map_or(0, |m| m.bag_total)
}

// The brain for a match we're playing, created on first use.
fn brain_for<'a>(db: &RemoteTables, st: &'a mut State, match_id: u64) -> Option<&'a mut Brain> {
    let me = st.my_bot_id?;
    balance_in(db, st, match_id)?;
    Some(st.brains.entry(match_id).or_insert_with(|| {
        let players = db.match_participant().iter().filter(|p| p.match_id == match_id).count();
        let theta = Theta::load(&theta_path()).unwrap_or_else(|e| {
            log!("no theta at {} ({e}); using defaults", theta_path());
            Theta::default()
        });
        let mut brain = Brain::new(WordIndex::shared(), players, me, theta);
        // The rollouts' rival bids and leaf values, when present (theta-search.txt uses both).
        if let Some(m) = data(&market_path(), Market::load) {
            brain.set_market(m);
        }
        if let Some(l) = data(&leave_path(), Leave::load) {
            brain.set_leave(l);
        }
        brain
    }))
}

// Bid on an open auction, or raise our bid if we can now afford more (say,
// after a word paid out mid-auction).
fn try_bid(db: &RemoteTables, reducers: &RemoteReducers, st: &mut State, a: &Auction) {
    let Some(balance) = balance_in(db, st, a.match_id) else { return };
    let ms_left = (a.closes_at.to_micros_since_unix_epoch() - Timestamp::now().to_micros_since_unix_epoch()) / 1000;
    if ms_left < MIN_TIME_LEFT_MS {
        return;
    }
    let (rack, left) = (rack_for_match(db, a.match_id), letters_left(db, a.match_id));
    let letter = code(a.letter.as_bytes()[0]);
    let Some(brain) = brain_for(db, st, a.match_id) else { return };
    let started = std::time::Instant::now();
    let amount = brain.bid(letter, &rack, balance as f64, left);
    if started.elapsed() > Duration::from_millis(200) {
        log!("slow bid: {} ms", started.elapsed().as_millis());
    }
    let alpha = brain.alpha;
    if amount <= st.bids.get(&a.id).copied().unwrap_or(0) {
        return;
    }
    st.bids.insert(a.id, amount);
    let _ = reducers.submit_bid_then(a.id, amount, log_failure(format!("bid on auction {}", a.id)));
    log!("bid {amount} on '{}' (match {}, auction {}, alpha {alpha:.2})", a.letter, a.match_id, a.id);
}

// Re-bid on the match's open auction, if any.
fn rebid(db: &RemoteTables, reducers: &RemoteReducers, st: &mut State, match_id: u64) {
    for a in db.auction().iter().filter(|a| a.match_id == match_id && a.status == AuctionStatus::Open) {
        try_bid(db, reducers, st, &a);
    }
}

// Play a word if the brain wants to. One word in flight per match; when it
// lands, look again, so the endgame can dump several words in one auction.
fn try_play_word(db: &RemoteTables, reducers: &RemoteReducers, st: &mut State, match_id: u64) {
    if st.playing.contains(&match_id) || !is_running(db, match_id) {
        return;
    }
    let Some(balance) = balance_in(db, st, match_id) else { return };
    let (rack, left) = (rack_for_match(db, match_id), letters_left(db, match_id));
    let Some(brain) = brain_for(db, st, match_id) else { return };
    let Some(word) = brain.word(&rack, balance as f64, left) else { return };
    let word = spell(&word);
    log!("match {match_id}: playing '{word}'");
    st.playing.insert(match_id);
    let what = format!("playing '{word}'");
    let _ = reducers.submit_word_then(match_id, word, move |ctx, res| {
        log_failure(what)(ctx, res);
        let mut st = state();
        st.playing.remove(&match_id);
        try_play_word(&ctx.db, &ctx.reducers, &mut st, match_id);
        rebid(&ctx.db, &ctx.reducers, &mut st, match_id);
    });
}

fn join_lobby(db: &RemoteTables, reducers: &RemoteReducers, st: &State) {
    let Some(me) = st.my_bot_id else { return };
    if db.match_participant().iter().any(|p| p.bot_id == me && is_running(db, p.match_id)) {
        return;
    }
    let open = db.lobby().iter().find(|l| l.status == LobbyStatus::Open);
    if open.is_some_and(|l| db.lobby_member().iter().any(|m| m.lobby_id == l.id && m.bot_id == me)) {
        return;
    }
    log!("joining lobby");
    let _ = reducers.join_lobby_then(log_failure("joining the lobby".into()));
}

fn bootstrap_activity(db: &RemoteTables, reducers: &RemoteReducers, st: &mut State) {
    if let Some(id) = env("TOURNAMENT_ID").and_then(|t| t.parse().ok()) {
        let _ = reducers.register_for_tournament_then(id, log_failure(format!("registering for tournament {id}")));
    }
    join_lobby(db, reducers, st);
    for a in db.auction().iter().filter(|a| a.status == AuctionStatus::Open) {
        try_bid(db, reducers, st, &a);
    }
    let me = st.my_bot_id;
    for p in db.match_participant().iter().filter(|p| Some(p.bot_id) == me) {
        try_play_word(db, reducers, st, p.match_id);
    }
}

fn subscribe_as_bot(conn: &impl DbContext<DbView = RemoteTables, Reducers = RemoteReducers, SubscriptionBuilder = spacetimedb_sdk::__codegen::SubscriptionBuilder<RemoteModule>>, st: &mut State) {
    let Some(me) = st.my_bot_id else { return };
    if st.subscribed_as_bot {
        return;
    }
    st.subscribed_as_bot = true;
    // Recent enough to cover a whole running match (~70 s), so a restart
    // mid-match can rebuild its bookkeeping.
    let ago = |s: u64| {
        let t = Timestamp::from_micros_since_unix_epoch(Timestamp::now().to_micros_since_unix_epoch() - s as i64 * 1_000_000);
        t.to_rfc3339().unwrap()
    };
    let (since, recent) = (ago(120), ago(5));
    let ours = format!("JOIN match_participant mp ON t.match_id = mp.match_id WHERE mp.bot_id = {me}");
    conn.subscription_builder()
        .on_applied(move |ctx| {
            let name = ctx.db.bot().id().find(&me).map_or("?".into(), |b| b.name);
            log!("acting as bot '{name}' (id {me})");
            bootstrap_activity(&ctx.db, &ctx.reducers, &mut state());
        })
        .on_error(|_, e| log!("subscription failed: {e}"))
        .subscribe(vec![
            "SELECT * FROM my_rack".to_string(),
            format!("SELECT t.* FROM match_participant t {ours}"),
            format!("SELECT t.* FROM match_state t JOIN match_participant mp ON t.id = mp.match_id WHERE mp.bot_id = {me}"),
            format!("SELECT t.* FROM auction t {ours} AND t.closes_at > '{recent}'"),
            format!("SELECT t.* FROM auction_result t {ours} AND t.closed_at > '{since}'"),
            format!("SELECT t.* FROM word_play t {ours} AND t.played_at > '{since}'"),
            format!("SELECT l.* FROM lobby l JOIN lobby_member lm ON l.id = lm.lobby_id WHERE lm.bot_id = {me}"),
            format!("SELECT * FROM lobby_member WHERE bot_id = {me}"),
            format!("SELECT * FROM bot WHERE id = {me}"),
        ]);
}

fn is_replay(event: &Event<Reducer>) -> bool {
    matches!(event, Event::SubscribeApplied)
}

fn on_connect(conn: &DbConnection, identity: Identity, token: &str) {
    state().my_identity = Some(identity);
    if let Err(e) = std::fs::write(token_path(), token) {
        log!("couldn't save token: {e}");
    }
    log!("connected as {}", identity.to_hex());

    // Subscribe only to what the bot needs. Phase 1 finds our bot id; phase 2
    // is scoped to it. (Subscribing to all tables pulls the server's whole
    // history and stalls the bot for seconds.)
    conn.subscription_builder()
        .on_applied(|ctx| {
            let mut st = state();
            st.my_bot_id = resolve_my_bot_id(&ctx.db, &st);
            if st.my_bot_id.is_some() {
                return subscribe_as_bot(ctx, &mut st);
            }
            let Some(nonce) = env("BOT_NONCE") else {
                log!("no BotCredential for this token. Set BOT_TOKEN or BOT_NONCE and re-run.");
                std::process::exit(1);
            };
            log!("claiming credential with nonce…");
            let _ = ctx.reducers.claim_credential_then(nonce, log_failure("claiming the credential".into()));
            // bot_credential's on_insert picks it up; give up if it doesn't come.
            std::thread::spawn(|| {
                std::thread::sleep(Duration::from_secs(5));
                if state().my_bot_id.is_none() {
                    log!("couldn't claim credential. Bad / expired nonce?");
                    std::process::exit(1);
                }
            });
        })
        .subscribe(format!("SELECT * FROM bot_credential WHERE identity = 0x{}", identity.to_hex()));

    conn.db.bot_credential().on_insert(|ctx, _| {
        let mut st = state();
        if st.my_bot_id.is_none() {
            st.my_bot_id = resolve_my_bot_id(&ctx.db, &st);
            subscribe_as_bot(ctx, &mut st);
        }
    });
    conn.db.auction().on_insert(|ctx, a| {
        let mut st = state();
        let letter = code(a.letter.as_bytes()[0]);
        if let Some(b) = brain_for(&ctx.db, &mut st, a.match_id) {
            b.note_auction(a.id, letter);
        }
        // Replayed rows are handled by bootstrap_activity once they're all in.
        if is_replay(&ctx.event) || a.status != AuctionStatus::Open {
            return;
        }
        try_bid(&ctx.db, &ctx.reducers, &mut st, a);
        // The bag shrinks every auction, which can turn a held rack into a
        // play, so look again even if the rack didn't change.
        try_play_word(&ctx.db, &ctx.reducers, &mut st, a.match_id);
    });
    conn.db.auction_result().on_insert(|ctx, r| {
        let mut st = state();
        let balance = balance_in(&ctx.db, &st, r.match_id).unwrap_or(0);
        let left = letters_left(&ctx.db, r.match_id);
        let me = st.my_bot_id;
        let Some(brain) = brain_for(&ctx.db, &mut st, r.match_id) else { return };
        brain.note_result(r.auction_id, code(r.letter.as_bytes()[0]), r.winner_bot_id, r.paid as f64, balance as f64, left);
        if !is_replay(&ctx.event) {
            let who = match r.winner_bot_id {
                Some(w) if Some(w) == me => "us".to_string(),
                Some(w) => format!("bot {w}"),
                None => "none".into(),
            };
            log!("match {} '{}' → {who} paid {}", r.match_id, r.letter, r.paid);
        }
    });
    conn.db.word_play().on_insert(|ctx, w| {
        let mut st = state();
        let word: Vec<u8> = w.word.bytes().map(code).collect();
        if let Some(b) = brain_for(&ctx.db, &mut st, w.match_id) {
            b.note_word(w.id, w.bot_id, &word);
        }
    });
    conn.db.my_rack().on_insert(|ctx, h| {
        if !is_replay(&ctx.event) {
            try_play_word(&ctx.db, &ctx.reducers, &mut state(), h.match_id);
        }
    });
    conn.db.my_rack().on_update(|ctx, _, h| try_play_word(&ctx.db, &ctx.reducers, &mut state(), h.match_id));
    conn.db.match_participant().on_update(|ctx, old, new| {
        let mut st = state();
        // A word paid out: we may afford a higher bid on the open auction.
        if Some(new.bot_id) == st.my_bot_id && new.balance > old.balance {
            rebid(&ctx.db, &ctx.reducers, &mut st, new.match_id);
        }
    });
    // When a match this bot was in ends, hop back into the lobby.
    conn.db.match_state().on_update(|ctx, old, new| {
        let mut st = state();
        if st.my_bot_id.is_none() || old.status == MatchStatus::Ended || new.status != MatchStatus::Ended {
            return;
        }
        st.brains.remove(&new.id);
        if balance_in(&ctx.db, &st, new.id).is_some() {
            let me = st.my_bot_id.unwrap();
            if let Some(p) = ctx.db.match_participant().iter().find(|p| p.match_id == new.id && p.bot_id == me) {
                log!("match {} ended with score {}; rejoining lobby", new.id, p.score);
            }
            join_lobby(&ctx.db, &ctx.reducers, &st);
        }
    });
}

fn main() {
    let host = env("STDB_HOST").unwrap_or_else(|| "https://maincloud.spacetimedb.com".into());
    let db_name = env("STDB_DB").unwrap_or_else(|| "scrabblebot".into());
    // Build the word index now rather than on the first auction.
    WordIndex::shared();
    // The Rust SDK doesn't reconnect by itself: on a disconnect, connect again
    // and resubscribe. Brains survive, so a match in progress carries on.
    loop {
        log!("connecting to {db_name} at {host}");
        let token = env("BOT_TOKEN").or_else(|| std::fs::read_to_string(token_path()).ok().map(|t| t.trim().to_string()));
        let (tx, rx) = mpsc::channel::<()>();
        let tx2 = tx.clone();
        let conn = DbConnection::builder()
            .with_uri(host.as_str())
            .with_database_name(db_name.as_str())
            .with_token(token)
            .on_connect(on_connect)
            .on_connect_error(move |_, e| {
                log!("connect error: {e}");
                let _ = tx2.send(());
            })
            .on_disconnect(move |_, e| {
                log!("disconnected{}", e.map_or(String::new(), |e| format!(": {e}")));
                let _ = tx.send(());
            })
            .build();
        match conn {
            Ok(conn) => {
                let _thread = conn.run_threaded();
                let _ = rx.recv();
            }
            Err(e) => log!("couldn't connect: {e}"),
        }
        {
            let mut st = state();
            st.subscribed_as_bot = false;
            st.playing.clear();
        }
        std::thread::sleep(Duration::from_secs(2));
    }
}
