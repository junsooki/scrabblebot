#!/bin/sh
# Place of Dwight (bot 98) in every match it finished on the adaptive build.
cd "$(dirname "$0")"
tail -n +2106 dwight.log | grep "ended with score" | awk '{print $3}' | while read m; do
  spacetime sql scrabblebot "SELECT bot_id, score FROM match_participant WHERE match_id = $m" 2>/dev/null | tail -n +3 |
    awk -v m=$m '{s[$1]=$3} END {me=s[98]; p=1; top=0; for (b in s) { if (b!=98 && s[b]>me) p++; if (b!=98 && s[b]>top) top=s[b] } print m, "Dwight", me, "best rival", top, "place", p}'
done
