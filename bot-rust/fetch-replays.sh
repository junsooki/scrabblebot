#!/bin/sh
# Download public match history from the live server for `cargo run --bin replay`.
# Usage: ./fetch-replays.sh <first match id> <last match id>
set -e
FROM=${1:?first match id}
TO=${2:?last match id}
OUT=$(dirname "$0")/replays
mkdir -p "$OUT"
q() { spacetime sql --server "${STDB_SERVER:-maincloud}" "${STDB_DB:-scrabblebot}" "$1" 2>/dev/null; }
W="match_id >= $FROM AND match_id <= $TO"
q "SELECT match_id, bot_id, score FROM match_participant WHERE $W" > "$OUT/participants.txt"
q "SELECT auction_id, match_id, letter, winner_bot_id, paid, closed_at FROM auction_result WHERE $W" > "$OUT/results.txt"
q "SELECT auction_id, top_bid FROM auction_top_bid_archive WHERE $W" > "$OUT/top_bids.txt"
q "SELECT match_id, bot_id, word, played_at FROM word_play WHERE $W" > "$OUT/words.txt"
wc -l "$OUT"/*.txt
