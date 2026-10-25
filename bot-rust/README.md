# Rust bot

A Rust port of `bot-starter`, plus tools to measure and tune it against the
live field offline.

| Command | What it does |
|---|---|
| `cargo run --release` | The bot. Same env vars as the TS starter (`BOT_NAME`, `BOT_TOKEN` or `BOT_NONCE`, `STDB_HOST`, `STDB_DB`, `TOURNAMENT_ID`), plus `THETA` (default `theta.txt`). |
| `./fetch-replays.sh 0 999999999` | Download the public match history the server still keeps into `replays/`. |
| `cargo run --release --bin replay -- --theta theta.txt` | Score a strategy by putting it in every seat of every real match (see `src/replay.rs`). |
| `cargo run --release --bin tune` | Tune `theta.txt` continuously against those replays; keeps changes that also win on held-out matches, and pulls new matches every 30 min. |
| `cargo test --release` | Strategy checks, including a decision-speed budget. |

The bot reads `theta.txt` at the start of every match, so a running bot picks
up whatever the tuner has learned.

## Trying it on a local server

```sh
spacetime start --listen-addr 127.0.0.1:3200 --in-memory --data-dir /tmp/stdb &
spacetime publish --server http://127.0.0.1:3200 --module-path ../spacetimedb -y scrabblebot-fake
# from outside the repo, so the CLI doesn't pick up spacetime.local.json:
spacetime call --server http://127.0.0.1:3200 scrabblebot-fake create_team '"fake-team"' '"rusty"'
spacetime call --server http://127.0.0.1:3200 scrabblebot-fake mint_credential_nonce
spacetime sql --server http://127.0.0.1:3200 scrabblebot-fake "SELECT code FROM my_nonces"
STDB_HOST=http://127.0.0.1:3200 STDB_DB=scrabblebot-fake BOT_NAME=rusty BOT_NONCE=<code> cargo run --release
```

Lobbies pad with the module's sim bots, which barely bid, so this checks the
client works, not how strong the strategy is. Use `replay` for that.

`src/module_bindings/` is generated:
`spacetime generate --lang rust --out-dir src/module_bindings --module-path ../spacetimedb`.
