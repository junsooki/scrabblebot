// A learned rack value, in the style of Maven/Quackle "superleaves": what a
// rack is worth in future points, as a sum of table entries a rack lights
// up. It costs about a hundred lookups instead of a dictionary scan.
//
// Features, one set per stage of the match: each letter held at least once,
// twice and three times; each pair of different letters held together (the
// synergy that makes ING or ERS better than its letters); the rack's size
// and vowel count; and coins. Fitted by ridge regression to the points a
// seat really went on to score in replays (`cargo run --bin learn`).

use crate::strategy::Rack;

const STAGES: usize = 4;
const SINGLES: usize = 26 * 3;
const PAIRS: usize = 26 * 25 / 2;
const SIZES: usize = 20;
const VOWELS: usize = 10;
// Bias, singles, pairs, size, vowels, coins.
const BLOCK: usize = 1 + SINGLES + PAIRS + SIZES + VOWELS + 1;
pub const FEATURES: usize = STAGES * BLOCK;

fn stage(left: u32) -> usize {
    match left {
        46.. => 0,
        26..=45 => 1,
        11..=25 => 2,
        _ => 3,
    }
}

fn pair(j: usize, k: usize) -> usize {
    j * 26 - j * (j + 1) / 2 + (k - j - 1)
}

// The features a rack lights up, as (index, value).
pub fn features(rack: &Rack, left: u32, balance: f64, out: &mut Vec<(u32, f32)>) {
    out.clear();
    let base = stage(left) * BLOCK;
    let mut push = |i: usize, v: f32| out.push(((base + i) as u32, v));
    push(0, 1.0);
    let mut size = 0;
    let mut vowels = 0;
    for k in 0..26 {
        let c = rack[k] as usize;
        size += c;
        if [0, 4, 8, 14, 20].contains(&k) {
            vowels += c;
        }
        for n in 0..c.min(3) {
            push(1 + k * 3 + n, 1.0);
        }
        if c > 0 {
            for j in 0..k {
                if rack[j] > 0 {
                    push(1 + SINGLES + pair(j, k), 1.0);
                }
            }
        }
    }
    push(1 + SINGLES + PAIRS + size.min(SIZES - 1), 1.0);
    push(1 + SINGLES + PAIRS + SIZES + vowels.min(VOWELS - 1), 1.0);
    push(BLOCK - 1, (balance / 100.0) as f32);
}

pub struct Leave {
    w: Vec<f64>,
}

impl Leave {
    pub fn value(&self, rack: &Rack, left: u32, balance: f64) -> f64 {
        let mut f = Vec::with_capacity(400);
        features(rack, left, balance, &mut f);
        f.iter().map(|&(i, v)| self.w[i as usize] * v as f64).sum()
    }

    // Ridge regression, (XᵀX + λI) w = Xᵀy, by conjugate gradient on the
    // sparse rows. The bias features go unpenalized in effect: λ is small
    // next to the thousands of samples each one sees.
    pub fn fit(rows: &[Vec<(u32, f32)>], y: &[f64], lambda: f64, iterations: usize) -> Leave {
        let n = FEATURES;
        let apply = |v: &[f64]| -> Vec<f64> {
            let mut out: Vec<f64> = v.iter().map(|x| lambda * x).collect();
            for r in rows {
                let dot: f64 = r.iter().map(|&(i, x)| v[i as usize] * x as f64).sum();
                for &(i, x) in r {
                    out[i as usize] += dot * x as f64;
                }
            }
            out
        };
        let mut b = vec![0.0; n];
        for (r, &t) in rows.iter().zip(y) {
            for &(i, x) in r {
                b[i as usize] += t * x as f64;
            }
        }
        let mut w = vec![0.0; n];
        let mut r = b.clone();
        let mut p = r.clone();
        let mut rs: f64 = r.iter().map(|x| x * x).sum();
        for _ in 0..iterations {
            let ap = apply(&p);
            let alpha = rs / p.iter().zip(&ap).map(|(a, b)| a * b).sum::<f64>().max(1e-12);
            for i in 0..n {
                w[i] += alpha * p[i];
                r[i] -= alpha * ap[i];
            }
            let next: f64 = r.iter().map(|x| x * x).sum();
            if next < 1e-9 {
                break;
            }
            for i in 0..n {
                p[i] = r[i] + next / rs * p[i];
            }
            rs = next;
        }
        Leave { w }
    }

    pub fn save(&self, path: &str) -> std::io::Result<()> {
        let text: String = self.w.iter().map(|w| format!("{w}\n")).collect();
        let tmp = format!("{path}.tmp");
        std::fs::write(&tmp, text)?;
        std::fs::rename(tmp, path)
    }

    pub fn load(path: &str) -> std::io::Result<Leave> {
        let w: Vec<f64> = std::fs::read_to_string(path)?.lines().filter_map(|l| l.trim().parse().ok()).collect();
        if w.len() != FEATURES {
            return Err(std::io::Error::other(format!("{path}: {} weights, expected {FEATURES}", w.len())));
        }
        Ok(Leave { w })
    }
}
