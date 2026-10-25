// Monte Carlo valuation for bids, the auction version of Macondo's
// simulation: play the rest of the match out many times and see what a tile
// (or a coin) is worth by how much more it scores.
//
// Each rollout draws the letters still to come from the unseen ones, and
// for each the best rival bid from the real distribution for that letter
// (the Market, built from the server's match history). A fast policy plays
// it: it buys a letter that some target word still needs when the rival bid
// is under its reserve price and it can afford it, and plays the best target
// word as soon as one is complete, getting its score back as coins. Every
// scenario of one decision shares the same rollouts, so their difference is
// mostly signal.

use crate::leave::Leave;
use crate::strategy::{Rack, WordIndex, LETTER_VALUE};
use std::collections::BinaryHeap;

// SplitMix64: small, fast and good enough for rollouts.
pub struct Rng(pub u64);
impl Rng {
    pub fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e3779b97f4a7c15);
        let mut x = self.0;
        x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
        x ^ (x >> 31)
    }
    pub fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
    pub fn unit(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }
}

const QUANTILES: usize = 21; // 0%, 5%, ..., 100%

// The best rival bid on each letter, as quantiles of what the field bid.
#[derive(Clone)]
pub struct Market {
    q: [[f64; QUANTILES]; 26],
}

impl Market {
    // From the top bids seen on each letter.
    pub fn from_bids(mut bids: Vec<Vec<f64>>) -> Market {
        let mut q = [[0.0; QUANTILES]; 26];
        for (k, b) in bids.iter_mut().enumerate() {
            if b.is_empty() {
                continue;
            }
            b.sort_by(f64::total_cmp);
            for (j, slot) in q[k].iter_mut().enumerate() {
                *slot = b[(j * (b.len() - 1)) / (QUANTILES - 1)];
            }
        }
        Market { q }
    }

    // The bid at quantile p of letter k's distribution.
    pub fn quantile(&self, k: usize, p: f64) -> f64 {
        let x = p.clamp(0.0, 1.0) * (QUANTILES - 1) as f64;
        let j = (x as usize).min(QUANTILES - 2);
        let f = x - j as f64;
        self.q[k][j] * (1.0 - f) + self.q[k][j + 1] * f
    }

    // One line per letter: the letter, then its quantiles.
    pub fn save(&self, path: &str) -> std::io::Result<()> {
        let mut out = String::new();
        for k in 0..26 {
            out.push((b'A' + k as u8) as char);
            for v in self.q[k] {
                out += &format!(" {v}");
            }
            out.push('\n');
        }
        let tmp = format!("{path}.tmp");
        std::fs::write(&tmp, out)?;
        std::fs::rename(tmp, path)
    }

    pub fn load(path: &str) -> std::io::Result<Market> {
        let mut q = [[0.0; QUANTILES]; 26];
        for line in std::fs::read_to_string(path)?.lines() {
            let mut parts = line.split_whitespace();
            let Some(k) = parts.next().and_then(|l| l.bytes().next()).map(|b| (b - b'A') as usize) else { continue };
            for (slot, v) in q[k.min(25)].iter_mut().zip(parts) {
                *slot = v.parse().unwrap_or(0.0);
            }
        }
        Ok(Market { q })
    }
}

pub struct Target {
    counts: Rack,
    score: f64,
}

// The best long words within `max_missing` letters of `rack`: the ones a
// rollout works towards. Ranked by score, discounted by what's missing.
pub fn targets(idx: &WordIndex, rack: &Rack, min_len: u8, max_missing: u32, limit: usize) -> Vec<Target> {
    struct Ranked(f64, usize);
    impl PartialEq for Ranked {
        fn eq(&self, o: &Self) -> bool {
            self.0 == o.0
        }
    }
    impl Eq for Ranked {}
    impl PartialOrd for Ranked {
        fn partial_cmp(&self, o: &Self) -> Option<std::cmp::Ordering> {
            Some(self.cmp(o))
        }
    }
    impl Ord for Ranked {
        fn cmp(&self, o: &Self) -> std::cmp::Ordering {
            o.0.total_cmp(&self.0) // min-heap: the worst kept pops first
        }
    }
    let total: u32 = rack.iter().map(|&c| c as u32).sum();
    let mask = (0..26).filter(|&k| rack[k] > 0).fold(0u32, |m, k| m | 1 << k);
    let mut heap = BinaryHeap::with_capacity(limit + 1);
    let (from, to) = idx.length_range(min_len as usize, (total + max_missing) as usize);
    for i in from..to {
        if (idx.mask(i) & !mask).count_ones() > max_missing {
            continue;
        }
        let c = idx.counts_of(i);
        let missing: u32 = (0..26).map(|k| c[k].saturating_sub(rack[k]) as u32).sum();
        if missing > max_missing {
            continue;
        }
        let rank = idx.score[i] as f64 * 0.8f64.powi(missing as i32);
        if heap.len() == limit && heap.peek().is_some_and(|w: &Ranked| rank <= w.0) {
            continue;
        }
        heap.push(Ranked(rank, i));
        if heap.len() > limit {
            heap.pop();
        }
    }
    heap.into_iter()
        .map(|Ranked(_, i)| {
            let mut counts = [0u8; 26];
            counts.copy_from_slice(idx.counts_of(i));
            Target { counts, score: idx.score[i] as f64 }
        })
        .collect()
}

// The rest of a match: the letters to come and the best rival bid on each.
pub struct Future {
    letters: Vec<u8>,
    rival: Vec<f64>,
}

pub fn futures(rng: &mut Rng, market: &Market, unseen: &[f64; 26], auctions: usize, n: usize) -> Vec<Future> {
    let total: f64 = unseen.iter().sum();
    (0..n)
        .map(|_| {
            let mut letters = Vec::with_capacity(auctions);
            let mut rival = Vec::with_capacity(auctions);
            for _ in 0..auctions {
                let mut x = rng.unit() * total;
                let mut k = 0;
                while k < 25 && x >= unseen[k] {
                    x -= unseen[k];
                    k += 1;
                }
                letters.push(k as u8);
                rival.push(market.quantile(k, rng.unit()));
            }
            Future { letters, rival }
        })
        .collect()
}

// Points the policy scores in one rollout from here. Tiles still held at the
// end count `dump` points per point of face value (short words in the
// final dump).
// With `leaf`, the rollout stops short of the end and the leave table
// prices what it holds there (rack and coins, `left` auctions still to go).
pub fn rollout(targets: &[Target], rack: &Rack, balance: f64, future: &Future, reserve: &[f64; 26], dump: f64, leaf: Option<(&Leave, u32)>) -> f64 {
    let mut rack = *rack;
    let mut balance = balance;
    let mut score = 0.0;
    let mut alive: Vec<bool> = vec![true; targets.len()];
    play_done(targets, &mut alive, &mut rack, &mut balance, &mut score, None);
    for (&x, &rival) in future.letters.iter().zip(&future.rival) {
        let k = x as usize;
        let price = rival.max(1.0);
        if rival >= reserve[k] || price > balance {
            continue;
        }
        let wanted = targets.iter().zip(&alive).any(|(t, &a)| a && t.counts[k] > rack[k]);
        if !wanted {
            continue;
        }
        balance -= price;
        rack[k] += 1;
        play_done(targets, &mut alive, &mut rack, &mut balance, &mut score, Some(k));
    }
    if let Some((leave, left)) = leaf {
        return score + leave.value(&rack, left, balance);
    }
    let leftover: f64 = (0..26).map(|m| rack[m] as f64 * LETTER_VALUE[m] as f64).sum();
    score + dump * leftover
}

// Play the best target the rack completes, if any; with `using`, only one
// that uses that letter (the one just bought) can be newly complete.
fn play_done(targets: &[Target], alive: &mut [bool], rack: &mut Rack, balance: &mut f64, score: &mut f64, using: Option<usize>) {
    let done = targets
        .iter()
        .enumerate()
        .filter(|(j, t)| alive[*j] && using.is_none_or(|k| t.counts[k] > 0) && (0..26).all(|m| t.counts[m] <= rack[m]))
        .max_by(|a, b| a.1.score.total_cmp(&b.1.score));
    if let Some((j, t)) = done {
        for m in 0..26 {
            rack[m] -= t.counts[m];
        }
        *score += t.score;
        *balance += t.score;
        alive[j] = false;
    }
}

#[allow(clippy::too_many_arguments)]
pub fn mean_rollout(targets: &[Target], rack: &Rack, balance: f64, futures: &[Future], reserve: &[f64; 26], dump: f64, leaf: Option<(&Leave, u32)>) -> f64 {
    futures.iter().map(|f| rollout(targets, rack, balance, f, reserve, dump, leaf)).sum::<f64>() / futures.len().max(1) as f64
}
