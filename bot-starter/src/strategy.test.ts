// Checks for the default strategy. Run with `npm test`.

import assert from "node:assert/strict";
import * as fs from "node:fs";
import { test } from "node:test";
import { chooseWord } from "./strategy.js";

const rack = (letters: string) => {
  const r = new Map<string, number>();
  for (const c of letters) r.set(c, (r.get(c) ?? 0) + 1);
  return r;
};

// At the end of the match chooseWord plays whatever scores best.
const endgame = { myBalance: 0, lettersLeft: 0, players: 6 };

test("chooseWord picks the longest word the rack can spell", () => {
  const dictionary = ["TACT", "CAT", "AT"];
  assert.equal(chooseWord({ myRack: rack("TACT"), dictionary, ...endgame }), "TACT");
  assert.equal(chooseWord({ myRack: rack("CATX"), dictionary, ...endgame }), "CAT");
  assert.equal(chooseWord({ myRack: rack("CX"), dictionary, ...endgame }), null);
});

test("chooseWord prefers the higher score over the longer word", () => {
  // QUIZ: 22 * 1.5 = 33; QUOTA: 14 * 2 = 28.
  const dictionary = ["QUOTA", "QUIZ"];
  assert.equal(chooseWord({ myRack: rack("QUIZOTA"), dictionary, ...endgame }), "QUIZ");
});

test("chooseWord holds short words mid-match but plays 7+ letters", () => {
  const dictionary = ["RETAINS", "STAIR"];
  const midMatch = { myBalance: 50, lettersLeft: 40, players: 6 };
  assert.equal(chooseWord({ myRack: rack("STAIRQ"), dictionary, ...midMatch }), null);
  assert.equal(chooseWord({ myRack: rack("RETAINS"), dictionary, ...midMatch }), "RETAINS");
});

// The bot can't see a match move on while it searches, so a slow search
// means bids on closed auctions and words from letters it already spent.
test("chooseWord searches the whole wordlist well within an auction", () => {
  const dictionary = fs
    .readFileSync(
      new URL("../../spacetimedb/wordlist.txt", import.meta.url),
      "utf8",
    )
    .split("\n")
    .map((l) => l.trim().toUpperCase())
    .filter((l) => l.length >= 2)
    .sort((a, b) => b.length - a.length);
  // The largest rack an admin can set, with almost nothing to spell.
  const myRack = rack("BCDFGHJKLMNPQRSTVWXZ");
  const start = performance.now();
  chooseWord({ myRack, dictionary, ...endgame });
  assert.ok(performance.now() - start < 100, "a search took 100 ms or more");
});
