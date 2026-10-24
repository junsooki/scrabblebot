// Checks for the strategy. Run with `npm test`.

import assert from "node:assert/strict";
import * as fs from "node:fs";
import { test } from "node:test";
import { MatchBrain } from "./strategy.js";

const rack = (letters: string) => {
  const r = new Map<string, number>();
  for (const c of letters) r.set(c, (r.get(c) ?? 0) + 1);
  return r;
};

test("word plays the best-scoring word at the end of the match", () => {
  const brain = new MatchBrain(["QUOTA", "QUIZ", "AT"], 6, "me");
  // QUIZ: 22 * 1.5 = 33; QUOTA: 14 * 2 = 28.
  assert.equal(brain.word(rack("QUIZOTA"), 50, 1), "QUIZ");
  assert.equal(brain.word(rack("CX"), 50, 1), null);
});

test("bid is 0 on the final tile, which can't be played", () => {
  const brain = new MatchBrain(["RETAINS"], 6, "me");
  assert.equal(brain.bid("S", rack("RETAIN"), 100, 0), 0);
});

test("bid pays more for a letter that completes a word", () => {
  const brain = new MatchBrain(["RETAINS", "QUIZ"], 6, "me");
  const helps = brain.bid("S", rack("RETAIN"), 100, 40);
  const useless = brain.bid("X", rack("RETAIN"), 100, 40);
  assert.ok(helps > useless, `${helps} <= ${useless}`);
  assert.ok(helps <= 100);
});

// The bot can't see a match move on while it decides, so slow decisions
// mean bids on closed auctions and words from letters it already spent.
test("decisions on the whole wordlist fit well within an auction", () => {
  const dictionary = fs
    .readFileSync(new URL("../../spacetimedb/wordlist.txt", import.meta.url), "utf8")
    .split("\n")
    .map((l) => l.trim().toUpperCase())
    .filter((l) => l.length >= 2);
  const brain = new MatchBrain(dictionary, 6, "me");
  brain.bid("E", rack("AINST"), 100, 60); // builds the index
  const start = performance.now();
  brain.bid("Q", rack("AEINRSTLU"), 100, 30);
  brain.word(rack("AEINRSTLU"), 100, 30);
  assert.ok(performance.now() - start < 100, "a decision took 100 ms or more");
});
