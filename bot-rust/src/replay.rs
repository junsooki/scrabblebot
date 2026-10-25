// Score a strategy against real matches from the live server.
//
// Download history with ./fetch-replays.sh, then
// For every match and every seat in it, the brain takes that seat's place.
// The match plays out as it really did around it: the same letters in the
// same order, and the same bids from the other five. In a Vickrey auction
// that's all a seat needs to know: the best rival bid is the recorded top
// bid, or the price paid when the seat itself won. The seat's real starting
// rack is hidden, so it gets five tiles drawn from the match's starting
// tiles. Rivals' scores stay as they were, so a tile the brain takes from a
// rival doesn't cost them anything here; that makes the score to beat a
// little harder than it really is.

use crate::leave::Leave;
use crate::sim::Market;
use crate::strategy::{code, word_score, Brain, Rack, Theta, WordIndex, FULL_BAG, LETTER_VALUE};
use std::collections::HashMap;
use std::sync::Arc;

pub struct Auction {
    id: u64,
    letter: u8,
    winner: Option<u64>,
    paid: i64,
    top: i64,
    closed_at: String,
}

pub struct Match {
    pub id: u64,
    pub seats: Vec<(u64, i64)>,               // bot id, real score
    pub auctions: Vec<Auction>,               // in order
    pub words: Vec<(String, u64, Vec<u8>)>,   // played at, bot id, word
}

fn rows(path: &str) -> Vec<Vec<String>> {
    let text = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("{path}: {e}. Run ./fetch-replays.sh first."));
    text.lines()
        .skip(2)
        .map(|l| l.split('|').map(|c| c.trim().trim_matches('"').to_string()).collect::<Vec<_>>())
        .filter(|c| c.len() > 1)
        .collect()
}

fn some(s: &str) -> Option<u64> {
    s.strip_prefix("(some = ")?.strip_suffix(')')?.parse().ok()
}

pub fn load(dir: &str, from: u64, to: u64) -> Vec<Match> {
    let mut by_id: HashMap<u64, Match> = HashMap::new();
    for r in rows(&format!("{dir}/participants.txt")) {
        let id: u64 = r[0].parse().unwrap();
        if id < from || id > to {
            continue;
        }
        by_id
            .entry(id)
            .or_insert_with(|| Match { id, seats: vec![], auctions: vec![], words: vec![] })
            .seats
            .push((r[1].parse().unwrap(), r[2].parse().unwrap()));
    }
    let tops: HashMap<u64, i64> =
        rows(&format!("{dir}/top_bids.txt")).iter().map(|r| (r[0].parse().unwrap(), r[1].parse().unwrap())).collect();
    for r in rows(&format!("{dir}/results.txt")) {
        let Some(m) = by_id.get_mut(&r[1].parse().unwrap()) else { continue };
        let id: u64 = r[0].parse().unwrap();
        let winner = some(&r[3]);
        let paid: i64 = r[4].parse().unwrap();
        let top = tops.get(&id).copied().unwrap_or(if winner.is_some() { paid } else { 0 });
        m.auctions.push(Auction { id, letter: code(r[2].as_bytes()[0]), winner, paid, top, closed_at: r[5].clone() });
    }
    for r in rows(&format!("{dir}/words.txt")) {
        let Some(m) = by_id.get_mut(&r[0].parse().unwrap()) else { continue };
        m.words.push((r[3].clone(), r[1].parse().unwrap(), r[2].bytes().map(code).collect()));
    }
    let mut matches: Vec<Match> = by_id
        .into_values()
        .filter(|m| m.seats.len() >= 2 && m.auctions.len() >= 60)
        .map(|mut m| {
            m.auctions.sort_by_key(|a| a.id);
            m.words.sort();
            m
        })
        .collect();
    matches.sort_by_key(|m| m.id);
    matches
}

pub use crate::sim::Rng;

// A decision point: the seat's state as an auction opened, and the points
// it scored from then to the end. What the leave table learns from.
pub struct State {
    pub rack: Rack,
    pub left: u32,
    pub balance: f64,
    pub heuristic: f64, // the brain's own valuation of the rack then
    pub future: f64,    // points scored from here on
}

pub struct Outcome {
    pub score: i64,
    pub won: u32,      // tiles won
    pub spent: f64,
    pub left_over: f64, // coins never spent
    pub unplayed: u32, // tiles held at the end
    pub long_points: i64, // points from 7+ letter words
    pub long_words: u32,
    pub long_by_stage: [u32; 3],
    pub won_by_stage: [u32; 3], // tiles won, same stages // 7+ words played with >45, 16-45 and <=15 auctions left
    pub real: i64,     // the real occupant's score
    pub best_rival: i64,
    pub place: u32, // 1 is first; ties share the better place
    pub states: Vec<State>, // filled when collecting
}

fn trace() -> bool {
    std::env::var("TRACE").is_ok()
}

// The learned rack value from `cargo run --bin learn`, if there is one
// (LEAVE=<file>, default leave.txt).
fn load_leave() -> Option<Arc<Leave>> {
    let path = std::env::var("LEAVE").unwrap_or_else(|_| concat!(env!("CARGO_MANIFEST_DIR"), "/leave.txt").into());
    Leave::load(&path).ok().map(Arc::new)
}

static LEAVE: std::sync::OnceLock<Option<Arc<Leave>>> = std::sync::OnceLock::new();

const SOMEONE: u64 = u64::MAX; // a rival who won a tile the real occupant had

#[allow(clippy::too_many_arguments)]
pub fn play_seat(index: &Arc<WordIndex>, market: &Arc<Market>, theta: &Theta, m: &Match, seat: usize, rng: &mut Rng, collect: bool) -> Outcome {
    let (me, real) = m.seats[seat];
    let mut brain = Brain::new(index.clone(), m.seats.len(), me, theta.clone());
    brain.set_market(market.clone());
    if let Some(leave) = LEAVE.get_or_init(load_leave) {
        brain.set_leave(leave.clone());
    }

    // Starting tiles: whatever of the bag was never auctioned.
    let mut pool: Vec<u8> = vec![];
    let mut auctioned = [0u8; 26];
    for a in &m.auctions {
        auctioned[a.letter as usize] += 1;
    }
    for k in 0..26 {
        for _ in auctioned[k]..FULL_BAG[k].max(auctioned[k]) {
            pool.push(k as u8);
        }
    }
    let mut rack: Rack = [0; 26];
    for _ in 0..5.min(pool.len()) {
        let j = rng.below(pool.len() as u64) as usize;
        rack[pool.swap_remove(j) as usize] += 1;
    }

    let mut balance = 100.0;
    let mut score = 0i64;
    let (mut won, mut spent, mut long_points, mut long_words) = (0u32, 0.0, 0i64, 0u32);
    let mut word_id = 0u64;
    let mut states = vec![];
    let mut rival_score: HashMap<u64, i64> = m.seats.iter().filter(|s| s.0 != me).map(|s| (s.0, 0)).collect();
    let mut rival_paid: HashMap<u64, i64> = HashMap::new();
    let mut long_by_stage = [0u32; 3];
    let mut won_by_stage = [0u32; 3];
    let mut next_word = 0;
    let n = m.auctions.len();
    for (i, a) in m.auctions.iter().enumerate() {
        let left = (n - i - 1) as u32;
        brain.note_auction(a.id, a.letter);
        if collect {
            let heuristic = brain.rack_value(&rack, left);
            states.push(State { rack, left, balance, heuristic, future: score as f64 });
        }
        while let Some(w) = brain.word(&rack, balance, left) {
            let mut base = 0;
            for &k in &w {
                rack[k as usize] -= 1;
                base += LETTER_VALUE[k as usize] as u32;
            }
            let reward = word_score(base, w.len()) as i64;
            score += reward;
            if trace() {
                eprintln!("   played {} for {reward}", crate::strategy::spell(&w));
            }
            if w.len() >= 7 {
                long_points += reward;
                long_words += 1;
                long_by_stage[if left > 45 { 0 } else if left > 15 { 1 } else { 2 }] += 1;
            }
            balance += reward as f64;
            word_id += 1;
            brain.note_word(u64::MAX - word_id, me, &w);
        }
        // The top-scoring rival so far, and whether it still has coins.
        let leader = rival_score.iter().max_by(|x, y| x.1.cmp(y.1)).map(|(b, _)| *b);
        brain.set_leader_solvent(leader.is_some_and(|b| 100 - rival_paid.get(&b).copied().unwrap_or(0) + rival_score[&b] >= 15));
        let bid = brain.bid(a.letter, &rack, balance, left);
        if trace() {
            eprintln!(
                "{left:2} {} bal {balance:4.0} rack {:<16} bid {bid:3} rival {:3}{}",
                (b'A' + a.letter) as char,
                crate::strategy::spell(&(0..26u8).flat_map(|k| std::iter::repeat_n(k, rack[k as usize] as usize)).collect::<Vec<_>>()),
                if a.winner == Some(me) { a.paid } else { a.top },
                if bid > (if a.winner == Some(me) { a.paid } else { a.top }) { "  WON" } else { "" }
            );
        }
        // The best bid among the other seats.
        let rival = if a.winner == Some(me) { a.paid } else { a.top };
        let we_win = bid > rival || (bid == rival && bid > 0 && rng.below(2) == 0);
        let (winner, paid) = if we_win {
            (Some(me), rival.max(1))
        } else if a.winner == Some(me) {
            // The real occupant won it; now the runner-up does, paying our bid.
            (Some(SOMEONE), bid.max(1))
        } else {
            (a.winner, a.paid.max(bid))
        };
        if we_win {
            balance -= paid as f64;
            spent += paid as f64;
            won += 1;
            won_by_stage[if left > 45 { 0 } else if left > 15 { 1 } else { 2 }] += 1;
            rack[a.letter as usize] += 1;
        }
        if let Some(w) = winner.filter(|&w| w != me) {
            *rival_paid.entry(w).or_insert(0) += paid;
        }
        brain.note_result(a.id, a.letter, winner, paid as f64, balance, left);
        // Rivals' words played before the next auction closed.
        let until = m.auctions.get(i + 1).map(|b| b.closed_at.as_str());
        while next_word < m.words.len() && until.is_none_or(|t| m.words[next_word].0.as_str() < t) {
            let (_, bot, w) = &m.words[next_word];
            if *bot != me {
                brain.note_word(next_word as u64, *bot, w);
                let base: u32 = w.iter().map(|&k| LETTER_VALUE[k as usize] as u32).sum();
                *rival_score.entry(*bot).or_insert(0) += word_score(base, w.len()) as i64;
            }
            next_word += 1;
        }
    }
    let rivals = m.seats.iter().enumerate().filter(|&(j, _)| j != seat).map(|(_, s)| s.1);
    let best_rival = rivals.clone().max().unwrap_or(0);
    let place = 1 + rivals.filter(|&r| r > score).count() as u32;
    let unplayed = rack.iter().map(|&c| c as u32).sum();
    for st in &mut states {
        st.future = score as f64 - st.future;
    }
    Outcome { score, won, spent, left_over: balance, unplayed, long_points, long_words, long_by_stage, won_by_stage, real, best_rival, place, states }
}


// Every seat of every match, in parallel. Each match draws its starting
// racks from its own seed, so two runs over the same matches are paired.
pub fn evaluate(index: &Arc<WordIndex>, market: &Arc<Market>, theta: &Theta, matches: &[Match]) -> Vec<Outcome> {
    evaluate_with(index, market, theta, matches, 0, false)
}

// `seed` varies the starting racks; `collect` records every decision point.
pub fn evaluate_with(index: &Arc<WordIndex>, market: &Arc<Market>, theta: &Theta, matches: &[Match], seed: u64, collect: bool) -> Vec<Outcome> {
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get());
    std::thread::scope(|s| {
        let handles: Vec<_> = (0..threads)
            .map(|t| {
                s.spawn(move || {
                    let mut out = vec![];
                    for m in matches.iter().skip(t).step_by(threads) {
                        let mut rng = Rng(m.id ^ seed.wrapping_mul(0x9e3779b97f4a7c15));
                        for seat in 0..m.seats.len() {
                            out.push(play_seat(index, market, theta, m, seat, &mut rng, collect));
                        }
                    }
                    out
                })
            })
            .collect();
        handles.into_iter().flat_map(|h| h.join().unwrap()).collect()
    })
}

// The best bid on each letter across the training matches, the rival bids
// rollouts draw from. Held-out matches stay out of it.
pub fn market(matches: &[Match]) -> Market {
    let mut bids = vec![vec![]; 26];
    for m in matches.iter().filter(|m| !is_held_out(m)) {
        for a in &m.auctions {
            bids[a.letter as usize].push(a.top as f64);
        }
    }
    Market::from_bids(bids)
}

// One match in three is held out from tuning, spread over the whole period.
pub fn is_held_out(m: &Match) -> bool {
    m.id % 3 == 0
}

pub struct Summary {
    pub seats: usize,
    pub score: f64,
    pub win: f64,
    // Placement points: 1 for first, 0 for last of six, so an average seat
    // gets 0.5. What the leaderboard's Plackett-Luce rating rewards.
    pub placement: f64,
}

pub fn summarize(outcomes: &[Outcome]) -> Summary {
    let n = outcomes.len().max(1) as f64;
    let mean = |f: &dyn Fn(&Outcome) -> f64| outcomes.iter().map(f).sum::<f64>() / n;
    Summary {
        seats: outcomes.len(),
        score: mean(&|o| o.score as f64),
        win: mean(&|o| (o.place == 1) as u8 as f64),
        placement: mean(&|o| (6.0 - o.place.min(6) as f64) / 5.0),
    }
}

impl std::fmt::Display for Summary {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "placement {:.4}, win rate {:.3}, score {:.1} ({} seats)", self.placement, self.win, self.score, self.seats)
    }
}
