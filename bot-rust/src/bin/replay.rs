// Score a strategy against real matches from the live server; see
// src/replay.rs for how. Download history with ./fetch-replays.sh, then
//   cargo run --release --bin replay -- [--theta FILE] [--from ID] [--to ID] [--split train|held-out] [--set name=value ...]
//
// --split picks the tuner's training or held-out matches.

use bot_rust::replay::{evaluate, is_held_out, load, market, summarize, Outcome};
use std::sync::Arc;
use bot_rust::strategy::{Theta, WordIndex};

fn main() {
    let mut theta = Theta::default();
    let (mut from, mut to) = (0u64, u64::MAX);
    let mut split = None;
    let mut dir = concat!(env!("CARGO_MANIFEST_DIR"), "/replays").to_string();
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        let mut val = || args.next().expect("missing value");
        match a.as_str() {
            "--theta" => theta = Theta::load(&val()).expect("reading theta"),
            "--from" => from = val().parse().unwrap(),
            "--to" => to = val().parse().unwrap(),
            "--dir" => dir = val(),
            "--split" => split = Some(val() == "held-out"),
            "--set" => {
                let kv = val();
                let (k, v) = kv.split_once('=').expect("--set name=value");
                theta.set(k, v.parse().expect("number"));
            }
            _ => panic!("unknown argument {a}"),
        }
    }
    let all = load(&dir, 0, u64::MAX);
    let market = Arc::new(market(&all));
    let mut matches: Vec<_> = all.into_iter().filter(|m| m.id >= from && m.id <= to).collect();
    if let Some(held_out) = split {
        matches.retain(|m| is_held_out(m) == held_out);
    }
    let started = std::time::Instant::now();
    let outcomes = evaluate(&WordIndex::shared(), &market, &theta, &matches);
    let mean = |f: &dyn Fn(&Outcome) -> f64| outcomes.iter().map(f).sum::<f64>() / outcomes.len() as f64;
    println!("{} matches in {:.1}s", matches.len(), started.elapsed().as_secs_f64());
    println!("{}", summarize(&outcomes));
    println!("real occupants scored {:.1}; beat the occupant {:.3}", mean(&|o| o.real as f64), mean(&|o| (o.score > o.real) as u8 as f64));
    println!(
        "tiles won {:.1}, spent {:.0}, coins left {:.0}, tiles unplayed {:.1}",
        mean(&|o| o.won as f64),
        mean(&|o| o.spent),
        mean(&|o| o.left_over),
        mean(&|o| o.unplayed as f64)
    );
    println!(
        "7+ words per match {:.2}, points per 7+ word {:.1}, share of points from 7+ {:.2}",
        mean(&|o| o.long_words as f64),
        mean(&|o| o.long_points as f64) / mean(&|o| o.long_words as f64),
        mean(&|o| o.long_points as f64) / mean(&|o| o.score as f64)
    );
}
