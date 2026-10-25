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

pub struct Outcome {
    pub score: i64,
    pub won: u32,      // tiles won
    pub spent: f64,
    pub left_over: f64, // coins never spent
    pub unplayed: u32, // tiles held at the end
    pub long_points: i64, // points from 7+ letter words
    pub long_words: u32,
    pub real: i64,     // the real occupant's score
    pub best_rival: i64,
    pub place: u32, // 1 is first; ties share the better place
}

fn trace() -> bool {
    std::env::var("TRACE").is_ok()
}

const SOMEONE: u64 = u64::MAX; // a rival who won a tile the real occupant had

pub fn play_seat(index: &Arc<WordIndex>, market: &Arc<Market>, theta: &Theta, m: &Match, seat: usize, rng: &mut Rng) -> Outcome {
    let (me, real) = m.seats[seat];
    let mut brain = Brain::new(index.clone(), m.seats.len(), me, theta.clone());
    brain.set_market(market.clone());

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
    let mut next_word = 0;
    let n = m.auctions.len();
    for (i, a) in m.auctions.iter().enumerate() {
        let left = (n - i - 1) as u32;
        brain.note_auction(a.id, a.letter);
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
            }
            balance += reward as f64;
            word_id += 1;
            brain.note_word(u64::MAX - word_id, me, &w);
        }
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
            rack[a.letter as usize] += 1;
        }
        brain.note_result(a.id, a.letter, winner, paid as f64, balance, left);
        // Rivals' words played before the next auction closed.
        let until = m.auctions.get(i + 1).map(|b| b.closed_at.as_str());
        while next_word < m.words.len() && until.is_none_or(|t| m.words[next_word].0.as_str() < t) {
            let (_, bot, w) = &m.words[next_word];
            if *bot != me {
                brain.note_word(next_word as u64, *bot, w);
            }
            next_word += 1;
        }
    }
    let rivals = m.seats.iter().enumerate().filter(|&(j, _)| j != seat).map(|(_, s)| s.1);
    let best_rival = rivals.clone().max().unwrap_or(0);
    let place = 1 + rivals.filter(|&r| r > score).count() as u32;
    let unplayed = rack.iter().map(|&c| c as u32).sum();
    Outcome { score, won, spent, left_over: balance, unplayed, long_points, long_words, real, best_rival, place }
}


// Every seat of every match, in parallel. Each match draws its starting
// racks from its own seed, so two runs over the same matches are paired.
pub fn evaluate(index: &Arc<WordIndex>, market: &Arc<Market>, theta: &Theta, matches: &[Match]) -> Vec<Outcome> {
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get());
    std::thread::scope(|s| {
        let handles: Vec<_> = (0..threads)
            .map(|t| {
                s.spawn(move || {
                    let mut out = vec![];
                    for m in matches.iter().skip(t).step_by(threads) {
                        let mut rng = Rng(m.id);
                        for seat in 0..m.seats.len() {
                            out.push(play_seat(index, market, theta, m, seat, &mut rng));
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
