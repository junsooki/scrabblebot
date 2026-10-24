// Checks for the default strategy. Run with `npm test`.

import assert from "node:assert/strict";
import * as fs from "node:fs";
import { test } from "node:test";
import {
  chooseWord,
  chooseWordReference,
  initializeStrategy,
  rackLeaveValue,
} from "./strategy.js";

const rack = (letters: string) => {
  const r = new Map<string, number>();
  for (const c of letters) r.set(c, (r.get(c) ?? 0) + 1);
  return r;
};

test("chooseWord picks the highest-scoring playable word", () => {
  const dictionary = ["TACT", "CAT", "AT"];
  assert.equal(chooseWord({ myRack: rack("TACT"), dictionary }), "TACT");
  assert.equal(chooseWord({ myRack: rack("CATX"), dictionary }), "CAT");
  assert.equal(chooseWord({ myRack: rack("CX"), dictionary }), null);
});

test("chooseWord can preserve high-score tiles for a later play", () => {
  const dictionary = ["CATERS", "JAZZ", "CAT"];
  assert.equal(chooseWord({ myRack: rack("JAZZCATERS"), dictionary }), "CATERS");
});

test("chooseWord trades immediate points for a valuable rack leave", () => {
  const dictionary = ["JAX", "ZA"];
  assert.equal(chooseWord({ myRack: rack("JAXZ"), dictionary }), "ZA");
});

test("rack leave equity penalizes duplicate tiles", () => {
  assert.ok(rackLeaveValue(rack("A")) > rackLeaveValue(rack("AA")));
});

test("native evaluator agrees with the TypeScript reference", () => {
  const dictionary = ["JAZZ", "CATERS", "JAX", "ZA", "CAT"];
  const ctx = { myRack: rack("JAZZCATERS"), dictionary };
  assert.equal(chooseWord(ctx), chooseWordReference(ctx));
});

// The bot can't see a match move on while it searches, so a slow search
// means bids on closed auctions and words from letters it already spent.
test("native chooseWord searches the whole wordlist well within an auction", () => {
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
  const ctx = { myRack, dictionary };
  const expected = chooseWordReference(ctx);
  initializeStrategy(dictionary);
  assert.equal(chooseWord(ctx), expected);

  const start = performance.now();
  chooseWord(ctx);
  assert.ok(performance.now() - start < 30, "native search took 30 ms or more");
});
