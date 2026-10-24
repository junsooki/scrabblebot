import { createRequire } from "node:module";

const require = createRequire(import.meta.url);
const native = require("../build/Release/scrabble_strategy.node") as {
  initializeDictionary(words: string[]): number;
  chooseWord(rack: string): string | null;
};

let initializedDictionary: string[] | null = null;

// Edit this file to implement your bidding and word-forming strategy.

// Standard Scrabble letter values — useful for valuation.
export const LETTER_VALUE: Record<string, number> = {
  A: 1, B: 3, C: 3, D: 2, E: 1, F: 4, G: 2, H: 4, I: 1, J: 8,
  K: 5, L: 1, M: 3, N: 1, O: 1, P: 3, Q: 10, R: 1, S: 1, T: 1,
  U: 1, V: 4, W: 4, X: 8, Y: 4, Z: 10,
};

export interface BidContext {
  letter: string;        // Letter being auctioned, e.g. "Q"
  myBalance: number;     // Current balance
  myRack: Map<string, number>; // Letters you currently own, letter -> count
}

/**
 * Decide how much to bid for `letter`. Return 0 to skip.
 * Default: bid the letter's face value plus a small premium if you don't have one yet.
 */
export function decideBid(ctx: BidContext): number {
  const value = LETTER_VALUE[ctx.letter] ?? 1;
  const have = ctx.myRack.get(ctx.letter) ?? 0;
  // Pay slightly above face value for letters you don't have yet.
  const want = have === 0 ? value + 2 : Math.max(1, value);
  return Math.min(want, Math.max(0, ctx.myBalance));
}

export interface WordContext {
  myRack: Map<string, number>;
  dictionary: string[]; // Uppercase list, shared at startup
}

/**
 * Try to form a word using letters from `myRack`.
 * Return the word (uppercase) to play, or null to skip.
 *
 * Quackle-style static equity: immediate score plus a value for the rack
 * left behind. This game has no board or fixed seven-tile refill, so leave
 * value is estimated from tile points, vowel/consonant balance, duplicates,
 * and whether Q has a usable partner.
 *
 * Keep this fast. The bot runs it on every rack change, and while it runs
 * the bot can't see the match move on: a search that takes longer than an
 * auction means bids on closed auctions and words from spent letters.
 */
/** Initialize the native evaluator once, before the bot starts receiving moves. */
export function initializeStrategy(dictionary: string[]): void {
  native.initializeDictionary(dictionary);
  initializedDictionary = dictionary;
}

/** Fast native implementation used by the live bot. */
export function chooseWord(ctx: WordContext): string | null {
  if (initializedDictionary !== ctx.dictionary) {
    initializeStrategy(ctx.dictionary);
  }
  let letters = "";
  for (const [letter, count] of ctx.myRack) {
    letters += letter.repeat(count);
  }
  return native.chooseWord(letters);
}

/** Pure TypeScript reference implementation, retained for parity tests. */
export function chooseWordReference(ctx: WordContext): string | null {
  let tiles = 0;
  for (const count of ctx.myRack.values()) tiles += count;

  let bestWord: string | null = null;
  let bestReward = -1;
  let bestEquity = Number.NEGATIVE_INFINITY;
  let bestLength = Number.POSITIVE_INFINITY;
  for (const word of ctx.dictionary) {
    if (word.length > tiles || !canSpell(word, ctx.myRack)) continue;

    const reward = wordReward(word);
    const leave = rackAfterWord(ctx.myRack, word);
    const equity = reward + rackLeaveValue(leave);

    // Maximize score plus the estimated future scoring power of the leave.
    // Prefer more points immediately when two plays have equal equity.
    if (
      equity > bestEquity ||
      (equity === bestEquity && reward > bestReward) ||
      (equity === bestEquity && reward === bestReward && word.length < bestLength)
    ) {
      bestWord = word;
      bestReward = reward;
      bestEquity = equity;
      bestLength = word.length;
    }
  }
  return bestWord;
}

function wordReward(word: string): number {
  const baseScore = [...word].reduce(
    (score, letter) => score + (LETTER_VALUE[letter] ?? 0),
    0,
  );
  const multiplier =
    word.length <= 3 ? 1 : word.length === 4 ? 1.5 :
      word.length === 5 ? 2 : word.length === 6 ? 2.5 : 3;
  return Math.floor(baseScore * multiplier);
}

function rackAfterWord(rack: Map<string, number>, word: string) {
  const leave = new Map(rack);
  for (const letter of word) {
    const remaining = (leave.get(letter) ?? 0) - 1;
    if (remaining <= 0) leave.delete(letter);
    else leave.set(letter, remaining);
  }
  return leave;
}

/** Approximate future word-making value of unplayed tiles, in score points. */
export function rackLeaveValue(rack: Map<string, number>): number {
  let tileCount = 0;
  let vowels = 0;
  let value = 0;
  let hasU = false;
  let hasI = false;
  let hasQ = false;

  for (const [letter, count] of rack) {
    tileCount += count;
    // A retained tile can earn its face value again, usually with a length
    // multiplier, so give it meaningful value while discounting uncertainty.
    value += count * (LETTER_VALUE[letter] ?? 0) * 1.2;
    if ("AEIOU".includes(letter)) vowels += count;
    if (letter === "U") hasU = count > 0;
    if (letter === "I") hasI = count > 0;
    if (letter === "Q") hasQ = count > 0;
    value -= Math.max(0, count - 1) * 0.8;
  }

  // Around 40% vowels tends to leave more ways to form future words.
  value -= Math.abs(vowels - tileCount * 0.4) * 1.5;
  if (hasQ && !hasU && !hasI) value -= 4;
  return value;
}

// Whether `rack` has every letter of `word`, counting repeats.
function canSpell(word: string, rack: Map<string, number>): boolean {
  const used: Record<string, number> = {};
  for (const c of word) {
    used[c] = (used[c] ?? 0) + 1;
    if (used[c] > (rack.get(c) ?? 0)) return false;
  }
  return true;
}
