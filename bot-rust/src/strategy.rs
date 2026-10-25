// The bidding and word-forming brain, a port of bot-starter/src/strategy.ts.
//
// One Brain per match. It keeps exact bag bookkeeping (what's still unseen),
// values the rack by the target words it can reach, bids that value
// converted to coins at a paced exchange rate, and decides when to cash in a
// word by comparing equity (score now + value of what's left) against
// holding. Background: Appel & Jacobson (1988) for the lexicon search idea,
// Maven/Quackle "equity" for play timing, and budget pacing from the
// repeated-auction literature (Balseiro & Gur).

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::sync::{Arc, OnceLock};

use crate::leave::Leave;
use crate::sim::{self, Market};

pub type Rack = [u8; 26];

// Standard Scrabble letter values, A to Z.
pub const LETTER_VALUE: [u8; 26] = [
    1, 3, 3, 2, 1, 4, 2, 4, 1, 8, 5, 1, 3, 1, 1, 3, 10, 1, 1, 1, 1, 4, 4, 8, 4, 10,
];

// The module's bag: standard Scrabble distribution, no blanks, 98 tiles.
pub const FULL_BAG: Rack = [
    9, 2, 2, 4, 12, 2, 3, 2, 9, 1, 1, 4, 2, 6, 8, 2, 1, 6, 4, 6, 4, 2, 2, 1, 2, 1,
];

const STARTING_RACK: usize = 5;

// Tunable parameters; tune them with `cargo run --release --bin replay`.
#[derive(Clone, Debug)]
pub struct Theta {
    pub alpha0: f64,     // starting coins paid per point of value
    pub alpha_min: f64,
    pub alpha_max: f64,
    pub eta: f64,        // pacing step
    pub rho0: f64,       // extra value of a reward that comes back as money to spend
    pub omega: [f64; 3], // our chance to win a given copy, by tier
    pub pi: [f64; 3],    // expected price when we win, by tier
    pub gamma: f64,      // weight of each further disjoint word
    pub stack: u32,      // most disjoint target words summed per rack
    pub flex: f64,       // weight of the alternatives to the best target word
    pub cash_target: f64, // coins below which cashing in a word gets a bonus
    pub boost: [f64; 3], // bid multiplier, by tier
    pub leave: f64,       // weight of the learned rack value (src/leave.rs) against the plans'
    pub sim: f64,         // 1: a tile's gain by Monte Carlo rollouts (src/sim.rs); 2: the whole bid
    pub sim_rollouts: usize,
    pub sim_missing: u32, // furthest target words a rollout works towards
    pub sim_targets: usize,
    pub sim_reserve: f64, // rollouts buy below this quantile of the rival bid
    pub sim_dump: f64,    // points per face-value point of tiles left at the end
    pub sim_coins: f64,   // extra coins used to price a coin
    pub sim_floor: f64,   // a coin is worth at least this × 1/alpha points
    pub sim_horizon: u32, // auctions a rollout plays before the leave table prices it (0: to the end)
    pub endgame: u32,     // solve the last this-many playable auctions exactly (needs a market)
    pub early_k: f64,     // bids are scaled by this while more than early_left auctions are left
    pub early_left: u32,
    pub max_missing: u32, // furthest target words considered
    pub top_n: usize,    // target words kept per rack
    pub k_end: u32,      // auctions left when endgame escalation starts
    pub kappa: f64,      // endgame escalation strength
    pub lottery: f64,    // floor bid on every letter early on
    pub play_k: usize,   // playable words compared when deciding to play
    pub epsilon: f64,    // play when equity is within this of holding
    pub min_play_len: u8, // shortest word worth playing before the endgame
    pub floor_k: f64,    // floor bid as a share of the letter's market price
}

impl Default for Theta {
    fn default() -> Self {
        Theta {
            alpha0: 1.7,
            alpha_min: 1.0,
            alpha_max: 4.0,
            eta: 0.08,
            rho0: 0.5,
            omega: [0.45, 0.4, 0.35],
            pi: [12.0, 19.0, 32.0],
            gamma: 0.7,
            stack: 2,
            flex: 0.0,
            cash_target: 0.0,
            boost: [1.0, 1.0, 1.0],
            leave: 0.0,
            sim: 0.0,
            sim_rollouts: 48,
            sim_missing: 4,
            sim_targets: 300,
            sim_reserve: 0.6,
            sim_dump: 0.5,
            sim_coins: 20.0,
            sim_floor: 1.0,
            sim_horizon: 0,
            endgame: 0,
            early_k: 1.0,
            early_left: 45,
            max_missing: 2,
            top_n: 64,
            k_end: 8,
            kappa: 1.5,
            lottery: 2.0,
            play_k: 6,
            epsilon: 1.0,
            min_play_len: 6,
            floor_k: 0.9,
        }
    }
}

impl Theta {
    // Every parameter by name, as tuned and saved.
    pub const NAMES: [&'static str; 40] = [
        "alpha0", "alpha_min", "alpha_max", "eta", "rho0", "omega0", "omega1", "omega2", "pi0", "pi1",
        "pi2", "gamma", "stack", "flex", "cash_target", "boost0", "boost1", "boost2", "leave", "sim", "sim_rollouts", "sim_missing", "sim_targets", "sim_reserve", "sim_dump", "sim_coins", "sim_floor", "sim_horizon", "endgame", "early_k", "early_left", "max_missing", "top_n", "k_end", "kappa", "lottery", "play_k", "epsilon",
        "min_play_len", "floor_k",
    ];

    pub fn get(&self, name: &str) -> f64 {
        match name {
            "alpha0" => self.alpha0,
            "alpha_min" => self.alpha_min,
            "alpha_max" => self.alpha_max,
            "eta" => self.eta,
            "rho0" => self.rho0,
            "omega0" => self.omega[0],
            "omega1" => self.omega[1],
            "omega2" => self.omega[2],
            "pi0" => self.pi[0],
            "pi1" => self.pi[1],
            "pi2" => self.pi[2],
            "gamma" => self.gamma,
            "stack" => self.stack as f64,
            "flex" => self.flex,
            "cash_target" => self.cash_target,
            "boost0" => self.boost[0],
            "boost1" => self.boost[1],
            "boost2" => self.boost[2],
            "leave" => self.leave,
            "sim" => self.sim,
            "sim_rollouts" => self.sim_rollouts as f64,
            "sim_missing" => self.sim_missing as f64,
            "sim_targets" => self.sim_targets as f64,
            "sim_reserve" => self.sim_reserve,
            "sim_dump" => self.sim_dump,
            "sim_coins" => self.sim_coins,
            "sim_floor" => self.sim_floor,
            "sim_horizon" => self.sim_horizon as f64,
            "endgame" => self.endgame as f64,
            "early_k" => self.early_k,
            "early_left" => self.early_left as f64,
            "max_missing" => self.max_missing as f64,
            "top_n" => self.top_n as f64,
            "k_end" => self.k_end as f64,
            "kappa" => self.kappa,
            "lottery" => self.lottery,
            "play_k" => self.play_k as f64,
            "epsilon" => self.epsilon,
            "min_play_len" => self.min_play_len as f64,
            "floor_k" => self.floor_k,
            _ => panic!("unknown parameter {name}"),
        }
    }

    // Integer parameters round; all stay in a range where the brain works.
    pub fn set(&mut self, name: &str, v: f64) {
        let int = |lo: f64, hi: f64| v.round().clamp(lo, hi);
        match name {
            "alpha0" => self.alpha0 = v.max(0.1),
            "alpha_min" => self.alpha_min = v.max(0.1),
            "alpha_max" => self.alpha_max = v.max(0.1),
            "eta" => self.eta = v.max(0.0),
            "rho0" => self.rho0 = v.max(0.0),
            "omega0" => self.omega[0] = v.clamp(0.01, 1.0),
            "omega1" => self.omega[1] = v.clamp(0.01, 1.0),
            "omega2" => self.omega[2] = v.clamp(0.01, 1.0),
            "pi0" => self.pi[0] = v.max(1.0),
            "pi1" => self.pi[1] = v.max(1.0),
            "pi2" => self.pi[2] = v.max(1.0),
            "gamma" => self.gamma = v.clamp(0.0, 1.0),
            "stack" => self.stack = int(1.0, 4.0) as u32,
            "flex" => self.flex = v.clamp(0.0, 2.0),
            "cash_target" => self.cash_target = v.clamp(0.0, 200.0),
            "boost0" => self.boost[0] = v.clamp(0.1, 5.0),
            "boost1" => self.boost[1] = v.clamp(0.1, 5.0),
            "boost2" => self.boost[2] = v.clamp(0.1, 5.0),
            "leave" => self.leave = v.clamp(0.0, 1.0),
            "sim" => self.sim = v,
            "sim_rollouts" => self.sim_rollouts = int(4.0, 1000.0) as usize,
            "sim_missing" => self.sim_missing = int(1.0, 7.0) as u32,
            "sim_targets" => self.sim_targets = int(10.0, 5000.0) as usize,
            "sim_reserve" => self.sim_reserve = v.clamp(0.0, 1.0),
            "sim_dump" => self.sim_dump = v.clamp(0.0, 3.0),
            "sim_coins" => self.sim_coins = v.clamp(1.0, 100.0),
            "sim_floor" => self.sim_floor = v.clamp(0.0, 5.0),
            "sim_horizon" => self.sim_horizon = int(0.0, 70.0) as u32,
            // Deeper takes over a second the first time, too slow for an auction.
            "endgame" => self.endgame = int(0.0, 2.0) as u32,
            "early_k" => self.early_k = v.clamp(0.0, 2.0),
            "early_left" => self.early_left = int(0.0, 67.0) as u32,
            "max_missing" => self.max_missing = int(1.0, 4.0) as u32,
            "top_n" => self.top_n = int(4.0, 256.0) as usize,
            "k_end" => self.k_end = int(1.0, 30.0) as u32,
            "kappa" => self.kappa = v.max(0.0),
            "lottery" => self.lottery = v.max(0.0),
            "play_k" => self.play_k = int(1.0, 32.0) as usize,
            "epsilon" => self.epsilon = v,
            "min_play_len" => self.min_play_len = int(2.0, 8.0) as u8,
            "floor_k" => self.floor_k = v.max(0.0),
            _ => panic!("unknown parameter {name}"),
        }
    }

    // `name = value` lines, as `save` writes them; missing names keep
    // defaults, and names this version doesn't know are skipped.
    pub fn load(path: &str) -> std::io::Result<Theta> {
        let mut theta = Theta::default();
        for line in std::fs::read_to_string(path)?.lines() {
            if let Some((k, v)) = line.split_once('=') {
                if let (true, Ok(v)) = (Theta::NAMES.contains(&k.trim()), v.trim().parse()) {
                    theta.set(k.trim(), v);
                }
            }
        }
        Ok(theta)
    }

    pub fn save(&self, path: &str, header: &str) -> std::io::Result<()> {
        let mut out = format!("# {header}\n");
        for name in Theta::NAMES {
            out += &format!("{name} = {}\n", self.get(name));
        }
        let tmp = format!("{path}.tmp");
        std::fs::write(&tmp, out)?;
        std::fs::rename(tmp, path)
    }
}

#[derive(Clone)]
struct Plan {
    ev: f64,    // expected value in points
    p: f64,     // chance of completing it
    s: f64,     // its score
    used: Rack, // rack letters it uses
}

pub struct Brain {
    pub alpha: f64,
    pub theta: Theta,
    players: usize,
    me: u64,
    index: Arc<WordIndex>,
    auctioned: HashMap<u64, u8>, // auction id -> letter code
    results: HashSet<u64>,
    words: HashSet<u64>,
    won: HashMap<u64, Rack>,    // bot id -> letters won
    played: HashMap<u64, Rack>, // bot id -> letters played
    prices: [f64; 3],
    income: f64, // money our current plans are expected to bring back
    rack: Rack,  // our actual rack at the last decision
    plan_cache: HashMap<Rack, Arc<Vec<Plan>>>,
    plan_cache_left: i64,
    market: Option<Arc<Market>>,
    leave: Option<Arc<Leave>>,
    final_scores: HashMap<Rack, f64>, // best_partition totals seen this match
}

impl Brain {
    pub fn new(index: Arc<WordIndex>, players: usize, me: u64, theta: Theta) -> Self {
        Brain {
            alpha: theta.alpha0,
            prices: theta.pi,
            theta,
            players,
            me,
            index,
            auctioned: HashMap::new(),
            results: HashSet::new(),
            words: HashSet::new(),
            won: HashMap::new(),
            played: HashMap::new(),
            income: 0.0,
            rack: [0; 26],
            plan_cache: HashMap::new(),
            plan_cache_left: -1,
            market: None,
            leave: None,
            final_scores: HashMap::new(),
        }
    }

    // The learned rack value (src/leave.rs) that theta.leave mixes in.
    pub fn set_leave(&mut self, leave: Arc<Leave>) {
        self.leave = Some(leave);
    }

    fn leave_mix(&self) -> f64 {
        if self.leave.is_some() { self.theta.leave } else { 0.0 }
    }

    // The rival bids rollouts draw from; without one, bids use the formula.
    pub fn set_market(&mut self, market: Arc<Market>) {
        self.market = Some(market);
    }

    // Bookkeeping. Each is idempotent, so replayed rows are harmless.

    pub fn note_auction(&mut self, auction_id: u64, letter: u8) {
        self.auctioned.insert(auction_id, letter);
    }

    pub fn note_result(
        &mut self,
        auction_id: u64,
        letter: u8,
        winner: Option<u64>,
        paid: f64,
        my_balance: f64,
        letters_left: u32,
    ) {
        self.note_auction(auction_id, letter);
        if !self.results.insert(auction_id) {
            return;
        }
        let Some(winner) = winner else { return };
        self.won.entry(winner).or_insert([0; 26])[letter as usize] += 1;
        let t = tier(letter as usize);
        self.prices[t] = 0.7 * self.theta.pi[t] + 0.3 * (0.8 * self.prices[t] + 0.2 * paid);
        // Pacing: nudge the exchange rate toward spending our money (plus what
        // our plans should bring back) evenly over the auctions left.
        let spent = if winner == self.me { paid } else { 0.0 };
        let target = (my_balance + spent + self.income) / (letters_left as f64 + 1.0).max(1.0);
        let th = &self.theta;
        self.alpha = (self.alpha * (th.eta * (target - spent) / target.max(1.0)).exp())
            .clamp(th.alpha_min, th.alpha_max);
    }

    pub fn note_word(&mut self, word_id: u64, bot_id: u64, word: &[u8]) {
        if !self.words.insert(word_id) {
            return;
        }
        let c = self.played.entry(bot_id).or_insert([0; 26]);
        for &ch in word {
            c[ch as usize] += 1;
        }
    }

    // Decisions.

    pub fn bid(&mut self, letter: u8, rack: &Rack, balance: f64, letters_left: u32) -> i64 {
        let bid = self.bid_unshaded(letter, rack, balance, letters_left);
        // The top late-game bots buy little while racks are still forming
        // and keep their coins for the tiles that finish words.
        if letters_left > self.theta.early_left {
            return (bid as f64 * self.theta.early_k).round() as i64;
        }
        bid
    }

    fn bid_unshaded(&mut self, letter: u8, rack: &Rack, balance: f64, letters_left: u32) -> i64 {
        // The final tile arrives as the match ends and can never be played.
        if letters_left == 0 || balance <= 0.0 {
            return 0;
        }
        // The last few auctions: solve them exactly instead.
        if let Some(market) = self.market.clone() {
            if self.theta.endgame > 0 && letters_left <= self.theta.endgame + 1 {
                return self.endgame_bid(&market, letter, rack, balance, letters_left);
            }
        }
        let market = self.market.clone().filter(|_| self.theta.sim > 0.5);
        if let Some(market) = &market {
            if self.theta.sim > 1.5 {
                return self.sim_bid(market, letter, rack, balance, letters_left).1;
            }
        }
        let mut r = *rack;
        self.set_rack(&r);
        let mix = self.leave_mix();
        let mut gain = 0.0;
        if mix < 1.0 {
            let before = self.value(&r, letters_left, true);
            r[letter as usize] += 1;
            gain = (1.0 - mix) * (self.value(&r, letters_left, false) - before);
            r[letter as usize] -= 1;
        }
        if let Some(leave) = self.leave.clone().filter(|_| mix > 0.0) {
            let mut with = r;
            with[letter as usize] += 1;
            gain += mix * (leave.value(&with, letters_left, balance) - leave.value(&r, letters_left, balance));
        }
        // sim 1: the points the tile adds, measured by rollouts.
        if let Some(market) = &market {
            gain = self.sim_bid(market, letter, rack, balance, letters_left).0;
        }
        let th = &self.theta;
        let mut a = self.alpha;
        if letters_left <= th.k_end {
            a *= 1.0 + th.kappa * (th.k_end - letters_left) as f64 / th.k_end as f64;
        }
        // Never bid far below the market (about 9 + 2.4 × letter value): an
        // empty-ish rack sees little gain in any one letter but still needs tiles.
        let market = (9.0 + 2.4 * LETTER_VALUE[letter as usize] as f64) * (a / th.alpha0);
        let mut bid = (gain * a * th.boost[tier(letter as usize)]).max(th.floor_k * market).round();
        // Money is worthless once the bag is empty: late, spend it on anything useful.
        if letters_left <= th.k_end && gain > 0.0 {
            bid = bid.max((balance / letters_left as f64).floor());
        }
        if letters_left > 15 {
            bid = bid.max(th.lottery);
        }
        bid.min(balance) as i64
    }

    pub fn word(&mut self, rack: &Rack, balance: f64, letters_left: u32) -> Option<Vec<u8>> {
        let r = *rack;
        self.set_rack(&r);
        let endgame = letters_left <= self.theta.k_end;
        let idx = self.index.clone();
        // Before the endgame, short words just burn tiles a long word needs.
        let playable: Vec<usize> = self
            .playable(&r)
            .into_iter()
            .filter(|&i| endgame || idx.lens[i] >= self.theta.min_play_len)
            .collect();
        let first = *playable.first()?;
        // Last chance: the tile won now is only playable while the final auction
        // runs, and nothing is playable after. Cash in the set of words that
        // scores most in total, its best word first.
        if letters_left <= 1 {
            let words = best_partition(&idx, &r, self.formable(&r));
            return Some(idx.words[words.first().copied().unwrap_or(first)].clone());
        }
        let rho = self.rho(letters_left);
        let mix = self.leave_mix();
        let leave = self.leave.clone().filter(|_| mix > 0.0);
        let hold = if mix < 1.0 { (1.0 - mix) * self.value(&r, letters_left, true) } else { 0.0 }
            + leave.as_ref().map_or(0.0, |l| mix * l.value(&r, letters_left, balance));
        // Short of money, a word's coins now are worth more than a better
        // word later: every auction we can't bid in is a tile lost.
        let cash_target = self.prices[0].max(self.theta.cash_target);
        let low_cash = (cash_target - balance).max(0.0) / self.alpha;
        let mut best = first;
        let mut best_equity = f64::NEG_INFINITY;
        for &i in &playable {
            let mut rest = r;
            let c = idx.counts(i);
            for k in 0..26 {
                rest[k] -= c[k];
            }
            let score = idx.score[i] as f64;
            let mut equity = 0.0;
            if mix < 1.0 {
                equity += (1.0 - mix) * (score * (1.0 + rho) + self.value(&rest, letters_left, false) + low_cash);
            }
            // The table counts the word's coins through the balance it sees.
            if let Some(l) = &leave {
                equity += mix * (score + l.value(&rest, letters_left, balance + score));
            }
            if equity > best_equity {
                best_equity = equity;
                best = i;
            }
        }
        (best_equity >= hold - self.theta.epsilon).then(|| idx.words[best].clone())
    }

    // The tile's value in coins by simulation: the points it adds over the
    // rest of the match, over the points a coin adds. Truthful, as a Vickrey
    // auction wants; late, when coins buy little, it bids up to everything.
    // Returns the points the tile adds and that converted to a bid.
    fn sim_bid(&mut self, market: &Market, letter: u8, rack: &Rack, balance: f64, letters_left: u32) -> (f64, i64) {
        let th = &self.theta;
        let mut with = *rack;
        with[letter as usize] += 1;
        self.rack = *rack;
        let targets = sim::targets(&self.index, &with, th.min_play_len, th.sim_missing, th.sim_targets);
        let unseen = self.usable(letters_left);
        let mut seed = letters_left as u64 ^ (balance as u64) << 8;
        for (k, &c) in rack.iter().enumerate() {
            seed = seed.wrapping_mul(31).wrapping_add(c as u64 * 26 + k as u64);
        }
        let mut rng = sim::Rng(seed);
        // The last auction's tile can never be played, so skip it.
        let mut auctions = letters_left.saturating_sub(1) as usize;
        // With a leave table, look sim_horizon auctions ahead and let it
        // price the rest.
        let leave = self.leave.clone().filter(|_| th.sim_horizon > 0);
        if leave.is_some() {
            auctions = auctions.min(th.sim_horizon as usize);
        }
        let leaf = leave.as_deref().map(|l| (l, letters_left - auctions as u32));
        let futures = sim::futures(&mut rng, market, &unseen, auctions, th.sim_rollouts);
        let mut reserve = [0.0; 26];
        for (k, r) in reserve.iter_mut().enumerate() {
            *r = market.quantile(k, th.sim_reserve);
        }
        let v = |r: &Rack, b: f64| sim::mean_rollout(&targets, r, b, &futures, &reserve, th.sim_dump, leaf);
        let without = v(rack, balance);
        let gain = v(&with, balance) - without;
        // Rollouts see one word ahead, so they undervalue money that would
        // fund the word after; the paced exchange rate floors a coin's worth.
        let per_coin = ((v(rack, balance + th.sim_coins) - without) / th.sim_coins).max(th.sim_floor / self.alpha);
        (gain, (gain / per_coin).clamp(0.0, balance).round() as i64)
    }

    // Expectimax over the rest of the match, once few enough auctions are
    // left: chance picks each letter (from the unseen ones) and the best
    // rival bid on it (from the market, at five quantiles); we decide whether
    // to buy at that price; at the end the rack scores its best set of words
    // (best_partition). Taking a tile exactly when it's worth more than its
    // price is what bidding our indifference price does in a Vickrey auction,
    // so the bid is the most we'd pay and still do as well as passing.
    fn endgame_bid(&mut self, market: &Market, letter: u8, rack: &Rack, balance: f64, letters_left: u32) -> i64 {
        // The final tile can never be played.
        if letters_left == 0 || balance <= 0.0 {
            return 0;
        }
        self.rack = *rack;
        let unseen = self.usable(letters_left);
        let total: f64 = unseen.iter().sum();
        let chances: Vec<(usize, f64, [f64; 5])> = (0..26)
            .filter(|&k| unseen[k] > 0.0)
            .map(|k| {
                let mut t = [0.0; 5];
                for (j, q) in t.iter_mut().enumerate() {
                    *q = market.quantile(k, 0.1 + 0.2 * j as f64).max(1.0);
                }
                (k, unseen[k] / total, t)
            })
            .collect();
        // Playable auctions after this one: all but the very last.
        let depth = letters_left - 1;
        let mut memo = HashMap::new();
        let mut with = *rack;
        with[letter as usize] += 1;
        let pass = self.expecti(&chances, rack, balance, depth, &mut memo);
        if self.expecti(&chances, &with, balance, depth, &mut memo) <= pass {
            return 0; // no use even free
        }
        // The most we can pay and still do as well as passing.
        let (mut lo, mut hi) = (0i64, balance as i64);
        while lo < hi {
            let mid = (lo + hi + 1) / 2;
            if self.expecti(&chances, &with, balance - mid as f64, depth, &mut memo) >= pass {
                lo = mid;
            } else {
                hi = mid - 1;
            }
        }
        lo
    }

    fn expecti(&mut self, chances: &[(usize, f64, [f64; 5])], rack: &Rack, balance: f64, depth: u32, memo: &mut HashMap<(Rack, i32, u32), f64>) -> f64 {
        if depth == 0 {
            return self.final_score(rack);
        }
        // Coins in steps of 2: close enough, and far fewer positions.
        let key = (*rack, (balance / 2.0).floor() as i32, depth);
        if let Some(&v) = memo.get(&key) {
            return v;
        }
        let pass = self.expecti(chances, rack, balance, depth - 1, memo);
        let mut v = 0.0;
        for &(k, w, prices) in chances {
            let mut with = *rack;
            with[k] += 1;
            let mut sum = 0.0;
            for t in prices {
                let take = if t <= balance { self.expecti(chances, &with, balance - t, depth - 1, memo) } else { pass };
                sum += take.max(pass);
            }
            v += w * sum / 5.0;
        }
        memo.insert(key, v);
        v
    }

    // The most the rack scores as disjoint words, remembered per rack for
    // the whole match: endgame searches keep meeting the same racks.
    fn final_score(&mut self, rack: &Rack) -> f64 {
        if let Some(&v) = self.final_scores.get(rack) {
            return v;
        }
        let idx = self.index.clone();
        let v = best_partition(&idx, rack, self.formable(rack)).iter().map(|&i| idx.score[i] as f64).sum();
        self.final_scores.insert(*rack, v);
        v
    }

    // The brain's valuation of a rack, in points.
    pub fn rack_value(&mut self, rack: &Rack, letters_left: u32) -> f64 {
        self.set_rack(rack);
        self.value(rack, letters_left, false)
    }

    // Valuation.

    fn set_rack(&mut self, r: &Rack) {
        if *r != self.rack {
            self.plan_cache.clear();
        }
        self.rack = *r;
    }

    fn rho(&self, letters_left: u32) -> f64 {
        self.theta.rho0 * ((letters_left as f64 - 4.0) / 40.0).clamp(0.0, 1.0)
    }

    // Unseen copies of each letter that we could still win and use.
    fn usable(&self, letters_left: u32) -> [f64; 26] {
        let mut auctioned = [0i32; 26];
        for &c in self.auctioned.values() {
            auctioned[c as usize] += 1;
        }
        let none = [0u8; 26];
        // Letters known to have come from starting racks: all of ours (what we
        // hold or played beyond what we won) and opponents' as they play them.
        let mut reveal = [0i32; 26];
        let my_won = self.won.get(&self.me).unwrap_or(&none);
        let my_played = self.played.get(&self.me).unwrap_or(&none);
        for k in 0..26 {
            reveal[k] = (self.rack[k] as i32 + my_played[k] as i32 - my_won[k] as i32).max(0);
        }
        let mut revealed = 0i32;
        for (bot, played) in &self.played {
            if *bot == self.me {
                continue;
            }
            let won = self.won.get(bot).unwrap_or(&none);
            for k in 0..26 {
                let from_start = (played[k] as i32 - won[k] as i32).max(0);
                revealed += from_start;
                reveal[k] += from_start;
            }
        }
        // Opponents' starting tiles we haven't seen yet are also unseen but not
        // in the bag, so only a share of the unseen copies will be auctioned.
        let hidden = ((self.players as i32 - 1) * STARTING_RACK as i32 - revealed).max(0);
        let share = (letters_left as f64 - 1.0).max(0.0) / (letters_left as f64 + hidden as f64).max(1.0);
        let mut unseen = [0f64; 26];
        for k in 0..26 {
            unseen[k] = (FULL_BAG[k] as i32 - auctioned[k] - reveal[k]).max(0) as f64 * share;
        }
        unseen
    }

    // V(rack): the summed value of the best set of disjoint target words
    // (up to theta.stack of them), the first at full weight and each further
    // one at gamma. With `track`, remember the money it should bring back.
    fn value(&mut self, rack: &Rack, letters_left: u32, track: bool) -> f64 {
        let plans = self.plans(rack, letters_left);
        let mut best = Combo::default();
        let mut path = Combo::default();
        combine(&plans, rack, 0, [0; 26], self.theta.stack, self.theta.gamma, 1.0, &mut path, &mut best);
        // Alternatives to the best word: each counts by the chance every
        // better one fell through, so a rack with many reachable words is
        // worth more than its single best shot. (It treats them as
        // independent, which they aren't when they share missing letters;
        // `flex` scales the bonus to make up for that.)
        let mut miss = 1.0;
        let mut flex = 0.0;
        for (j, pl) in plans.iter().enumerate() {
            if j > 0 {
                flex += miss * pl.ev;
            }
            miss *= 1.0 - pl.p;
        }
        best.value += self.theta.flex * flex;
        if track {
            self.income = best.income;
        }
        best.value
    }

    // The top_n target words within max_missing letters of `rack`, by expected
    // value in points: P(we win the missing letters) × score, minus what
    // they'd cost converted to points.
    fn plans(&mut self, rack: &Rack, letters_left: u32) -> Arc<Vec<Plan>> {
        if letters_left as i64 != self.plan_cache_left {
            self.plan_cache.clear();
            self.plan_cache_left = letters_left as i64;
        }
        if let Some(hit) = self.plan_cache.get(rack) {
            return hit.clone();
        }
        let th = &self.theta;
        let idx = &*self.index;
        let usable = self.usable(letters_left);
        let rho = self.rho(letters_left);
        let total: u32 = rack.iter().map(|&c| c as u32).sum();
        let mask = mask_of(rack);
        // Per letter: the chance of winning `short` more copies, by short.
        let mut tail = [[1.0f64; 5]; 26];
        for (k, t) in tail.iter_mut().enumerate() {
            for (short, v) in t.iter_mut().enumerate().skip(1) {
                *v = poisson_tail(usable[k] * th.omega[tier(k)], short as u32);
            }
        }
        // A min-heap on ev holding the top_n best so far.
        let mut top: BinaryHeap<Worst> = BinaryHeap::with_capacity(th.top_n + 1);
        for i in idx.start_at_len(total + th.max_missing)..idx.words.len() {
            if (idx.masks[i] & !mask).count_ones() > th.max_missing {
                continue;
            }
            let s = idx.score[i] as f64;
            // ev is at most s * (1 + rho), so skip what can't make the cut.
            if top.len() == th.top_n && s * (1.0 + rho) <= top.peek().map_or(0.0, |w| w.0.ev) {
                continue;
            }
            let letters = idx.letters(i);
            let missing: u32 = letters.iter().map(|&(k, n)| n.saturating_sub(rack[k as usize]) as u32).sum();
            if missing > th.max_missing {
                continue;
            }
            let mut p = 1.0;
            let mut cost = 0.0;
            for &(k, n) in letters {
                let short = n.saturating_sub(rack[k as usize]);
                if short > 0 {
                    p *= tail[k as usize][(short as usize).min(4)];
                    cost += short as f64 * self.prices[tier(k as usize)];
                }
            }
            let ev = if missing == 0 {
                s * (1.0 + rho)
            } else {
                p * s * (1.0 + rho / 2.0) - cost / self.alpha
            };
            if ev <= 0.0 {
                continue;
            }
            if top.len() == th.top_n && ev <= top.peek().map_or(0.0, |w| w.0.ev) {
                continue;
            }
            let c = idx.counts(i);
            let mut used = [0u8; 26];
            for k in 0..26 {
                used[k] = c[k].min(rack[k]);
            }
            top.push(Worst(Plan { ev, p, s, used }));
            if top.len() > th.top_n {
                top.pop();
            }
        }
        // Best first.
        let top: Vec<Plan> = top.into_sorted_vec().into_iter().map(|w| w.0).collect();
        let top = Arc::new(top);
        self.plan_cache.insert(*rack, top.clone());
        top
    }

    // Up to play_k playable words, best score first.
    fn playable(&self, rack: &Rack) -> Vec<usize> {
        let mut found = self.formable(rack);
        found.truncate(self.theta.play_k);
        found
    }

    // Every word the rack can spell, best score first.
    fn formable(&self, rack: &Rack) -> Vec<usize> {
        let idx = &*self.index;
        let total: u32 = rack.iter().map(|&c| c as u32).sum();
        let mask = mask_of(rack);
        let mut found: Vec<usize> = (idx.start_at_len(total)..idx.words.len())
            .filter(|&i| idx.masks[i] & !mask == 0 && formable(idx.counts(i), rack))
            .collect();
        found.sort_by(|&a, &b| idx.score[b].cmp(&idx.score[a]).then(idx.lens[b].cmp(&idx.lens[a])));
        found
    }
}

// The set of words, spelled from disjoint tiles of `rack`, with the highest
// total score: best-first search on a max-heap keyed by an upper bound (the
// score so far plus 3× the face value of every tile left, the most a tile
// can earn). A node that pops as "done" has a bound no other node beats, so
// it's optimal. `words` are the rack's formable words, best first; a node
// only adds words at or after its last one, so each set is reached once.
// Gives the best set found if the search runs past its node budget.
pub fn best_partition(idx: &WordIndex, rack: &Rack, words: Vec<usize>) -> Vec<usize> {
    struct Node {
        bound: u32,
        score: u32,
        done: bool,
        rest: Rack,
        from: usize,
        chosen: Vec<usize>,
    }
    impl PartialEq for Node {
        fn eq(&self, o: &Self) -> bool {
            self.cmp(o) == Ordering::Equal
        }
    }
    impl Eq for Node {}
    impl PartialOrd for Node {
        fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
            Some(self.cmp(o))
        }
    }
    impl Ord for Node {
        fn cmp(&self, o: &Self) -> Ordering {
            // Ties: finished sets first, then deeper ones.
            (self.bound, self.done, self.score).cmp(&(o.bound, o.done, o.score))
        }
    }
    let ceiling = |r: &Rack| (0..26).map(|k| 3 * r[k] as u32 * LETTER_VALUE[k] as u32).sum::<u32>();
    let mut heap = BinaryHeap::new();
    heap.push(Node { bound: ceiling(rack), score: 0, done: false, rest: *rack, from: 0, chosen: vec![] });
    let mut best: (u32, Vec<usize>) = (0, vec![]);
    let mut budget = 20_000;
    while let Some(node) = heap.pop() {
        if node.done {
            return node.chosen;
        }
        if node.score > best.0 {
            best = (node.score, node.chosen.clone());
        }
        budget -= 1;
        if budget == 0 {
            break;
        }
        for (j, &w) in words.iter().enumerate().skip(node.from) {
            let c = idx.counts(w);
            if !formable(c, &node.rest) {
                continue;
            }
            let mut rest = node.rest;
            for k in 0..26 {
                rest[k] -= c[k];
            }
            let score = node.score + idx.score[w] as u32;
            let mut chosen = node.chosen.clone();
            chosen.push(w);
            heap.push(Node { bound: score + ceiling(&rest), score, done: false, rest, from: j, chosen });
        }
        heap.push(Node { bound: node.score, done: true, ..node });
    }
    best.1
}

// Orders plans so a BinaryHeap pops the worst first.
struct Worst(Plan);
impl PartialEq for Worst {
    fn eq(&self, o: &Self) -> bool {
        self.cmp(o) == Ordering::Equal
    }
}
impl Eq for Worst {}
impl PartialOrd for Worst {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
        Some(self.cmp(o))
    }
}
impl Ord for Worst {
    fn cmp(&self, o: &Self) -> Ordering {
        o.0.ev.total_cmp(&self.0.ev)
    }
}

#[derive(Default, Clone, Copy)]
struct Combo {
    value: f64,
    income: f64,
}

// Depth-first over sets of plans whose rack letters fit together, taking
// plans in best-first order and pruning when even the best remaining plans
// at full weight can't beat the best set found.
#[allow(clippy::too_many_arguments)]
fn combine(plans: &[Plan], rack: &Rack, from: usize, used: Rack, left: u32, gamma: f64, weight: f64, path: &mut Combo, best: &mut Combo) {
    if path.value > best.value {
        *best = *path;
    }
    if left == 0 {
        return;
    }
    for j in from..plans.len() {
        let pj = &plans[j];
        if path.value + weight * pj.ev * left as f64 <= best.value {
            return; // plans are sorted best first, so no later one helps either
        }
        if !(0..26).all(|k| used[k] + pj.used[k] <= rack[k]) {
            continue;
        }
        let mut u = used;
        for k in 0..26 {
            u[k] += pj.used[k];
        }
        let saved = *path;
        path.value += weight * pj.ev;
        path.income += weight * pj.p * pj.s;
        combine(plans, rack, j + 1, u, left - 1, gamma, weight * gamma, path, best);
        *path = saved;
    }
}

fn formable(word: &[u8], rack: &Rack) -> bool {
    (0..26).all(|k| word[k] <= rack[k])
}

// P(X >= k) for X ~ Poisson(mu).
fn poisson_tail(mu: f64, k: u32) -> f64 {
    if mu <= 0.0 {
        return 0.0;
    }
    let mut term = (-mu).exp();
    let mut below = 0.0;
    for j in 0..k {
        below += term;
        term *= mu / (j + 1) as f64;
    }
    (1.0 - below).max(0.0)
}

// 0: 1-2 point letters, 1: B C F H M P V W Y, 2: J K Q X Z.
fn tier(k: usize) -> usize {
    match LETTER_VALUE[k] {
        0..=2 => 0,
        5.. => 2,
        _ => 1,
    }
}

fn mask_of(rack: &Rack) -> u32 {
    (0..26).filter(|&k| rack[k] > 0).fold(0, |m, k| m | 1 << k)
}

// The dictionary as letter signatures: per word its letter counts (flat,
// 26 per word) and a bitmask of the letters it uses, sorted longest first so
// a scan can skip words too long to reach.
pub struct WordIndex {
    pub words: Vec<Vec<u8>>, // letter codes 0-25
    counts: Vec<u8>,
    masks: Vec<u32>,
    pub score: Vec<u16>,
    pub lens: Vec<u8>,
    start: Vec<usize>,
    letters: Vec<(u8, u8)>, // per word its distinct letters and counts, flat
    letters_at: Vec<u32>,   // where each word's run starts in `letters`
}

impl WordIndex {
    pub fn new(dictionary: &str) -> Self {
        let mut words: Vec<Vec<u8>> = dictionary
            .lines()
            .map(|l| l.trim().to_ascii_uppercase())
            .filter(|l| l.len() >= 2 && l.bytes().all(|b| b.is_ascii_uppercase()))
            .map(|l| l.bytes().map(|b| b - b'A').collect())
            .collect();
        words.sort_by(|a, b| b.len().cmp(&a.len()));
        let n = words.len();
        let mut counts = vec![0u8; n * 26];
        let mut masks = vec![0u32; n];
        let mut score = vec![0u16; n];
        let mut lens = vec![0u8; n];
        for (i, w) in words.iter().enumerate() {
            let mut base = 0u32;
            for &k in w {
                counts[i * 26 + k as usize] += 1;
                masks[i] |= 1 << k;
                base += LETTER_VALUE[k as usize] as u32;
            }
            lens[i] = w.len() as u8;
            score[i] = word_score(base, w.len()) as u16;
        }
        let max_len = lens.first().copied().unwrap_or(0) as usize;
        let mut start = vec![0usize; max_len + 2];
        let mut i = n;
        for (len, s) in start.iter_mut().enumerate() {
            while i > 0 && lens[i - 1] as usize <= len {
                i -= 1;
            }
            *s = i;
        }
        let mut letters = vec![];
        let mut letters_at = Vec::with_capacity(n + 1);
        for i in 0..n {
            letters_at.push(letters.len() as u32);
            for k in 0..26 {
                if counts[i * 26 + k] > 0 {
                    letters.push((k as u8, counts[i * 26 + k]));
                }
            }
        }
        letters_at.push(letters.len() as u32);
        WordIndex { words, counts, masks, score, lens, start, letters, letters_at }
    }

    // The ENABLE wordlist the module validates words against.
    pub fn shared() -> Arc<WordIndex> {
        static INDEX: OnceLock<Arc<WordIndex>> = OnceLock::new();
        INDEX
            .get_or_init(|| Arc::new(WordIndex::new(include_str!("../../spacetimedb/wordlist.txt"))))
            .clone()
    }

    fn letters(&self, i: usize) -> &[(u8, u8)] {
        &self.letters[self.letters_at[i] as usize..self.letters_at[i + 1] as usize]
    }

    pub fn counts_of(&self, i: usize) -> &[u8] {
        self.counts(i)
    }

    pub fn mask(&self, i: usize) -> u32 {
        self.masks[i]
    }

    // Indices of the words from min_len to max_len letters long.
    pub fn length_range(&self, min_len: usize, max_len: usize) -> (usize, usize) {
        let from = self.start_at_len(max_len as u32);
        (from, self.start_at_len(min_len.saturating_sub(1) as u32).max(from))
    }

    fn counts(&self, i: usize) -> &[u8] {
        &self.counts[i * 26..i * 26 + 26]
    }

    // First word no longer than len.
    fn start_at_len(&self, len: u32) -> usize {
        let len = len as usize;
        if len + 2 > self.start.len() { 0 } else { self.start[len] }
    }
}

// Mirrors the module's reward: floor(base * length multiplier).
pub fn word_score(base: u32, len: usize) -> u32 {
    let (num, den) = match len {
        0..=3 => (1, 1),
        4 => (3, 2),
        5 => (2, 1),
        6 => (5, 2),
        _ => (3, 1),
    };
    base * num / den
}

pub fn code(c: u8) -> u8 {
    c.to_ascii_uppercase() - b'A'
}

pub fn spell(word: &[u8]) -> String {
    word.iter().map(|&k| (b'A' + k) as char).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rack(letters: &str) -> Rack {
        let mut r = [0; 26];
        for b in letters.bytes() {
            r[code(b) as usize] += 1;
        }
        r
    }

    fn brain(words: &str) -> Brain {
        Brain::new(Arc::new(WordIndex::new(words)), 6, 1, Theta::default())
    }

    #[test]
    fn word_plays_the_best_scoring_word_at_the_end_of_the_match() {
        // QUIZ: 22 * 1.5 = 33; QUOTA: 14 * 2 = 28.
        let mut b = brain("QUOTA\nQUIZ\nAT");
        assert_eq!(b.word(&rack("QUIZOTA"), 50.0, 1).map(|w| spell(&w)), Some("QUIZ".into()));
        assert_eq!(b.word(&rack("CX"), 50.0, 1), None);
    }

    #[test]
    fn the_final_dump_maximises_the_total_not_the_first_word() {
        // Greedy takes TOTAL (5 letters, 5 × 2 = 10) and is left with O and E.
        let idx = WordIndex::new("TOTAL\nLATTE\nOE");
        let best = best_partition(&idx, &rack("AELOTTO"), (0..idx.words.len()).collect());
        let total: u32 = best.iter().map(|&i| idx.score[i] as u32).sum();
        // TOTAL (10) + OE (2) = 12 beats LATTE (10) with O left over.
        assert_eq!(total, 12);
    }

    #[test]
    fn bid_is_0_on_the_final_tile_which_cant_be_played() {
        assert_eq!(brain("RETAINS").bid(code(b'S'), &rack("RETAIN"), 100.0, 0), 0);
    }

    #[test]
    fn bid_pays_more_for_a_letter_that_completes_a_word() {
        let mut b = brain("RETAINS\nQUIZ");
        let helps = b.bid(code(b'S'), &rack("RETAIN"), 100.0, 40);
        let useless = b.bid(code(b'X'), &rack("RETAIN"), 100.0, 40);
        assert!(helps > useless, "{helps} <= {useless}");
        assert!(helps <= 100);
    }

    // The bot can't see a match move on while it decides, so slow decisions
    // mean bids on closed auctions and words from letters it already spent.
    #[test]
    fn decisions_on_the_whole_wordlist_fit_well_within_an_auction() {
        let mut b = Brain::new(WordIndex::shared(), 6, 1, Theta::default());
        b.bid(code(b'E'), &rack("AINST"), 100.0, 60); // builds the index
        let start = std::time::Instant::now();
        b.bid(code(b'Q'), &rack("AEINRSTLU"), 100.0, 30);
        b.word(&rack("AEINRSTLU"), 100.0, 30);
        b.word(&rack("AEINRSTLUOCDMPH"), 100.0, 1);
        let ms = start.elapsed().as_millis();
        assert!(ms < 100, "decisions took {ms} ms");
    }
}

#[cfg(test)]
mod bench {
    use super::*;

    #[test]
    #[ignore]
    fn decision_cost() {
        let mut b = Brain::new(WordIndex::shared(), 6, 1, Theta::load("theta.txt").unwrap());
        let mut r = [0u8; 26];
        for c in b"AEINRSTLO" {
            r[code(*c) as usize] += 1;
        }
        b.bid(code(b'E'), &r, 100.0, 61); // builds the index
        let start = std::time::Instant::now();
        for left in (1..=60).cycle().take(std::env::var("BENCH_N").map_or(60, |n| n.parse().unwrap())) {
            b.bid(code(b'E'), &r, 100.0, left);
            b.word(&r, 100.0, left);
        }
        let n = std::env::var("BENCH_N").map_or(60.0, |n| n.parse().unwrap());
        eprintln!("{:.2} ms per bid + word", start.elapsed().as_secs_f64() * 1000.0 / n);
    }
}
