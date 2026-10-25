#!/bin/sh
# Download public match history from the live server for `cargo run --bin replay`.
# Usage: ./fetch-replays.sh <first match id> <last match id>
# Each file is downloaded beside the old one and only replaces it if the
# download worked and has rows, so a failed fetch never leaves empty data.
set -e
FROM=${1:?first match id}
TO=${2:?last match id}
OUT=$(dirname "$0")/replays
mkdir -p "$OUT"
W="match_id >= $FROM AND match_id <= $TO"
fetch() {
  spacetime sql --server "${STDB_SERVER:-maincloud}" "${STDB_DB:-scrabblebot}" "$2" > "$OUT/$1.new" 2>/dev/null
  [ "$(wc -l < "$OUT/$1.new")" -gt 2 ] || { echo "fetch of $1 came back empty; keeping the old file" >&2; rm -f "$OUT/$1.new"; exit 1; }
}
fetch participants.txt "SELECT match_id, bot_id, score FROM match_participant WHERE $W"
fetch results.txt "SELECT auction_id, match_id, letter, winner_bot_id, paid, closed_at FROM auction_result WHERE $W"
fetch top_bids.txt "SELECT auction_id, top_bid FROM auction_top_bid_archive WHERE $W"
fetch words.txt "SELECT match_id, bot_id, word, played_at FROM word_play WHERE $W"
for f in participants.txt results.txt top_bids.txt words.txt; do mv "$OUT/$f.new" "$OUT/$f"; done
wc -l "$OUT"/*.txt
