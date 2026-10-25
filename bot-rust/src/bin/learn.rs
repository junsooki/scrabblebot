// Fit the leave table (src/leave.rs) to replays of the current strategy.
//
//   cargo run --release --bin learn -- [--theta theta.txt] [--passes 3] [--out leave.txt]
//
// Plays every seat of the training matches `passes` times (different
// starting racks each time), records each decision point and the points
// scored after it, fits the table, and reports how well it predicts on
// held-out matches next to the brain's own rack valuation.

use bot_rust::leave::{features, Leave};
use bot_rust::replay::{evaluate_with, is_held_out, load, market, State};
use bot_rust::strategy::{Theta, WordIndex};
use std::sync::Arc;

fn states(outcomes: Vec<bot_rust::replay::Outcome>) -> Vec<State> {
    outcomes.into_iter().flat_map(|o| o.states).collect()
}

// R² of predictions against what happened.
fn r2(pred: &[f64], y: &[f64]) -> f64 {
    let mean = y.iter().sum::<f64>() / y.len() as f64;
    let total: f64 = y.iter().map(|v| (v - mean).powi(2)).sum();
    let resid: f64 = pred.iter().zip(y).map(|(p, v)| (v - p).powi(2)).sum();
    1.0 - resid / total
}

fn rows(states: &[State]) -> Vec<Vec<(u32, f32)>> {
    states
        .iter()
        .map(|s| {
            let mut f = vec![];
            features(&s.rack, s.left, s.balance, &mut f);
            f
        })
        .collect()
}

fn main() {
    let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/replays");
    let mut theta_path = concat!(env!("CARGO_MANIFEST_DIR"), "/theta.txt").to_string();
    let mut out = concat!(env!("CARGO_MANIFEST_DIR"), "/leave.txt").to_string();
    let mut passes = 3;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        let mut val = || args.next().expect("missing value");
        match a.as_str() {
            "--theta" => theta_path = val(),
            "--passes" => passes = val().parse().unwrap(),
            "--out" => out = val(),
            _ => panic!("unknown argument {a}"),
        }
    }
    let theta = Theta::load(&theta_path).unwrap_or_default();
    let all = load(dir, 0, u64::MAX);
    let rivals = Arc::new(market(&all));
    let (train, test): (Vec<_>, Vec<_>) = all.into_iter().partition(|m| !is_held_out(m));
    let index = WordIndex::shared();

    let started = std::time::Instant::now();
    let mut seen = vec![];
    for pass in 0..passes {
        seen.extend(states(evaluate_with(&index, &rivals, &theta, &train, pass + 1, true)));
    }
    let held = states(evaluate_with(&index, &rivals, &theta, &test, 0, true));
    println!("{} training and {} held-out decision points in {:.0}s", seen.len(), held.len(), started.elapsed().as_secs_f64());

    let y: Vec<f64> = seen.iter().map(|s| s.future).collect();
    let leave = Leave::fit(&rows(&seen), &y, 10.0, 200);
    leave.save(&out).unwrap();

    let hy: Vec<f64> = held.iter().map(|s| s.future).collect();
    let pred: Vec<f64> = held.iter().map(|s| leave.value(&s.rack, s.left, s.balance)).collect();
    // The brain's valuation, given the same chance: a line fitted per stage
    // on training points with its value and the coins as inputs. (Its raw
    // value doesn't count coins or the words after its plans.)
    let baseline = |s: &State| -> Vec<(u32, f32)> {
        let st = match s.left {
            46.. => 0,
            26..=45 => 1,
            11..=25 => 2,
            _ => 3,
        } * 3;
        vec![(st, 1.0), (st + 1, s.heuristic as f32), (st + 2, (s.balance / 100.0) as f32)]
    };
    let base_rows: Vec<_> = seen.iter().map(baseline).collect();
    let base = fit_small(&base_rows, &y, 12);
    let base_pred: Vec<f64> = held.iter().map(|s| baseline(s).iter().map(|&(i, x)| base[i as usize] * x as f64).sum()).collect();
    println!("held-out R² of future points: leave table {:.3}, brain's valuation {:.3}", r2(&pred, &hy), r2(&base_pred, &hy));
    println!("wrote {out}");
}

// Least squares for a handful of features, by normal equations.
fn fit_small(rows: &[Vec<(u32, f32)>], y: &[f64], n: usize) -> Vec<f64> {
    let mut a = vec![vec![0.0; n + 1]; n];
    for (r, &t) in rows.iter().zip(y) {
        for &(i, x) in r {
            for &(j, z) in r {
                a[i as usize][j as usize] += x as f64 * z as f64;
            }
            a[i as usize][n] += x as f64 * t;
        }
    }
    for (i, row) in a.iter_mut().enumerate() {
        row[i] += 1e-6;
    }
    // Gaussian elimination.
    for c in 0..n {
        let piv = (c..n).max_by(|&i, &j| a[i][c].abs().total_cmp(&a[j][c].abs())).unwrap();
        a.swap(c, piv);
        for r in 0..n {
            if r != c && a[c][c] != 0.0 {
                let f = a[r][c] / a[c][c];
                for k in c..=n {
                    a[r][k] -= f * a[c][k];
                }
            }
        }
    }
    (0..n).map(|i| if a[i][i] != 0.0 { a[i][n] / a[i][i] } else { 0.0 }).collect()
}
