// Learn the strategy's parameters offline, continuously, from real matches.
//
//   cargo run --release --bin tune -- [--theta theta.txt] [--refresh-minutes 30] [--matches N]
//
// --matches keeps only the newest N matches, for speed and to follow the
// field as it is now.
//
// A (1+1) evolution strategy: mutate a few parameters of the best theta so
// far, replay it in every seat of the training matches (see src/replay.rs),
// and keep it if its placement beats the best's on the training matches and
// then also on held-out matches it was never tuned on. Step sizes grow after
// a success and shrink after failures (Rechenberg's 1/5 rule). A candidate
// first plays half the training matches and stops there unless it beats
// the best on that half, which halves the cost of the many that don't. The best goes
// to the theta file after every improvement; the bot reads that file at the
// start of every match, so a running bot picks up what the tuner learns.
//
// Every --refresh-minutes it downloads the newest matches (./fetch-replays.sh)
// and re-scores the best on them, so it keeps up with how the field plays.
// Runs until stopped.

use bot_rust::replay::{evaluate, is_held_out, load, market, summarize, Match, Rng, Summary};
use std::sync::Arc;
use bot_rust::strategy::{Theta, WordIndex};
use std::io::Write;
use std::time::{Duration, Instant};

// Parameters worth tuning, with the scale of a typical step.
const TUNED: [(&str, f64); 31] = [
    ("alpha0", 0.3),
    ("alpha_min", 0.2),
    ("alpha_max", 0.5),
    ("eta", 0.03),
    ("rho0", 0.2),
    ("omega0", 0.08),
    ("omega1", 0.08),
    ("omega2", 0.08),
    ("pi0", 2.0),
    ("pi1", 3.0),
    ("pi2", 5.0),
    ("gamma", 0.15),
    ("stack", 1.0),
    ("k_end", 2.0),
    ("kappa", 0.3),
    ("lottery", 1.0),
    ("play_k", 2.0),
    ("epsilon", 2.0),
    ("min_play_len", 1.0),
    ("floor_k", 0.1),
    ("flex", 0.2),
    ("max_missing", 1.0),
    ("boost0", 0.15),
    ("boost1", 0.2),
    ("boost2", 0.3),
    ("sim_horizon", 2.0),
    ("sim_reserve", 0.08),
    ("sim_coins", 5.0),
    ("sim_missing", 1.0),
    ("sim_targets", 60.0),
    ("endgame", 1.0),
];

const DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/replays");

fn split(matches: Vec<Match>) -> (Vec<Match>, Vec<Match>) {
    let (mut train, mut test) = (vec![], vec![]);
    for m in matches {
        if is_held_out(&m) { test.push(m) } else { train.push(m) }
    }
    (train, test)
}

fn keep_newest(mut matches: Vec<Match>, n: usize) -> Vec<Match> {
    matches.drain(..matches.len().saturating_sub(n));
    matches
}

fn refresh() {
    let script = concat!(env!("CARGO_MANIFEST_DIR"), "/fetch-replays.sh");
    match std::process::Command::new(script).args(["0", "999999999"]).output() {
        Ok(out) if out.status.success() => {}
        Ok(out) => eprintln!("fetch-replays failed: {}", String::from_utf8_lossy(&out.stderr)),
        Err(e) => eprintln!("fetch-replays failed: {e}"),
    }
}

fn main() {
    let mut path = concat!(env!("CARGO_MANIFEST_DIR"), "/theta.txt").to_string();
    let mut refresh_every = Duration::from_secs(30 * 60);
    let mut newest = usize::MAX;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        let mut val = || args.next().expect("missing value");
        match a.as_str() {
            "--theta" => path = val(),
            "--refresh-minutes" => refresh_every = Duration::from_secs(val().parse::<u64>().unwrap() * 60),
            "--matches" => newest = val().parse().unwrap(),
            _ => panic!("unknown argument {a}"),
        }
    }
    let log_path = format!("{path}.log");
    // Where the bot finds the rival bids its rollouts draw from.
    let market_path = concat!(env!("CARGO_MANIFEST_DIR"), "/market.txt");
    let mut log = std::fs::OpenOptions::new().create(true).append(true).open(&log_path).unwrap();
    let mut say = |line: String| {
        println!("{line}");
        writeln!(log, "{line}").unwrap();
    };

    let index = WordIndex::shared();
    let mut best = Theta::load(&path).unwrap_or_default();
    let all = keep_newest(load(DIR, 0, u64::MAX), newest);
    let mut rivals = Arc::new(market(&all));
    rivals.save(&market_path).unwrap();
    let (mut train, mut test) = split(all);
    let score = |theta: &Theta, rivals: &Arc<_>, ms: &[Match]| -> Summary { summarize(&evaluate(&index, rivals, theta, ms)) };
    let mut best_half = score(&best, &rivals, &train[..train.len() / 2]);
    let mut best_train = score(&best, &rivals, &train);
    let mut best_test = score(&best, &rivals, &test);
    say(format!("start: {} train, {} held-out matches", train.len(), test.len()));
    say(format!("  train    {best_train}\n  held out {best_test}"));

    let mut rng = Rng(std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos() as u64);
    let gauss = |rng: &mut Rng| {
        // Box-Muller.
        let u = (rng.next() >> 11) as f64 / (1u64 << 53) as f64;
        let v = (rng.next() >> 11) as f64 / (1u64 << 53) as f64;
        (-2.0 * (u + 1e-12).ln()).sqrt() * (std::f64::consts::TAU * v).cos()
    };
    let mut sigma = 1.0;
    let mut last_refresh = Instant::now();
    for generation in 1.. {
        if last_refresh.elapsed() >= refresh_every {
            last_refresh = Instant::now();
            refresh();
            let all = keep_newest(load(DIR, 0, u64::MAX), newest);
            rivals = Arc::new(market(&all));
            rivals.save(&market_path).unwrap();
            (train, test) = split(all);
            best_half = score(&best, &rivals, &train[..train.len() / 2]);
            best_train = score(&best, &rivals, &train);
            best_test = score(&best, &rivals, &test);
            say(format!("refreshed: {} train, {} held-out matches", train.len(), test.len()));
            say(format!("  train    {best_train}\n  held out {best_test}"));
        }

        // Mutate one to three parameters.
        let mut cand = best.clone();
        let mut changed = vec![];
        for _ in 0..1 + rng.below(3) {
            let (name, step) = TUNED[rng.below(TUNED.len() as u64) as usize];
            cand.set(name, cand.get(name) + sigma * step * gauss(&mut rng));
            changed.push(format!("{name}={:.3}", cand.get(name)));
        }
        if TUNED.iter().all(|(n, _)| cand.get(n) == best.get(n)) {
            continue; // an integer step that rounded away
        }
        let half = &train[..train.len() / 2];
        let mut accepted = false;
        let first = score(&cand, &rivals, half);
        if first.placement <= best_half.placement {
            sigma = (sigma * 0.9_f64.powf(0.25)).clamp(0.2, 3.0);
            if generation % 10 == 0 {
                say(format!("gen {generation}: best still {best_test} (sigma {sigma:.2})"));
            }
            continue;
        }
        let t = score(&cand, &rivals, &train);
        if t.placement > best_train.placement {
            let h = score(&cand, &rivals, &test);
            if h.placement > best_test.placement {
                accepted = true;
                best = cand;
                best_half = first;
                best_train = t;
                best_test = h;
                say(format!("gen {generation}: kept {}", changed.join(" ")));
                say(format!("  train    {best_train}\n  held out {best_test}"));
                best.save(&path, &format!("held out: {best_test}")).unwrap();
            }
        }
        sigma = (sigma * if accepted { 1.5 } else { 0.9_f64.powf(0.25) }).clamp(0.2, 3.0);
        if !accepted && generation % 10 == 0 {
            say(format!("gen {generation}: best still {best_test} (sigma {sigma:.2})"));
        }
    }
}
