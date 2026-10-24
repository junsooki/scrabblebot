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
  dictionary: string[];
  lettersLeft: number;   // Tiles still in the bag
  players: number;       // Bots in the match
}

// A word we're up to this many letters short of still counts as reachable.
const MAX_MISSING = 2;
// How much a word is worth per letter it's still missing.
const MISSING_DISCOUNT = 0.5;
// Potential is the mean of this many best reachable words, so a letter that
// helps a runner-up word is still worth something.
const TOP_K = 5;

/**
 * Decide how much to bid for `letter`. Return 0 to skip.
 *
 * Auctions are Vickrey (winner pays the runner-up), so bid what the letter
 * is worth: how much it raises our potential, i.e. the discounted score of
 * the nearest words to our rack. On top of that, pace spending: money left
 * when the bag empties is worthless, so the fewer letters remain the more of
 * our balance each one gets.
 */
export function decideBid(ctx: BidContext): number {
  const index = indexFor(ctx.dictionary);
  const rack = rackCounts(ctx.myRack);
  const before = potential(index, rack);
  rack[code(ctx.letter)]++;
  const gain = potential(index, rack) - before;
  // Roughly the share of the remaining bag we can expect to win.
  const budget = (ctx.myBalance * ctx.players) / Math.max(ctx.players, ctx.lettersLeft);
  const bid = Math.round(gain + budget / 4);
  return Math.min(Math.max(1, bid), Math.max(0, ctx.myBalance));
}

export interface WordContext {
  myRack: Map<string, number>;
  dictionary: string[]; // Sorted uppercase list, shared at startup
  myBalance: number;
  lettersLeft: number;
  players: number;
}

/**
 * Pick a word to play from `myRack`, or null to hold.
 *
 * Long words pay up to 3x, so we hold tiles until we can spell a 7+ letter
 * word. We play earlier only when holding stops paying: at the end of the
 * match, when unplayed tiles are worthless, or when we're too broke to bid.
 *
 * Keep this fast. The bot runs it on every rack change, and while it runs
 * the bot can't see the match move on: a search that takes longer than an
 * auction means bids on closed auctions and words from spent letters.
 */
export function chooseWord(ctx: WordContext): string | null {
  const index = indexFor(ctx.dictionary);
  const rack = rackCounts(ctx.myRack);
  const mask = rackMask(rack);
  let best = -1;
  for (let i = 0; i < index.words.length; i++) {
    if (missing(index, i, rack, mask, 0) !== 0) continue;
    if (best < 0 || index.score[i] > index.score[best]) best = i;
  }
  if (best < 0) return null;
  const word = index.words[best];
  const endgame = ctx.lettersLeft <= ctx.players;
  const broke = ctx.myBalance < 10;
  return word.length >= 7 || endgame || broke ? word : null;
}

// The dictionary as letter signatures: for each word, its letter counts and
// a bitmask of which letters it uses. How far a word is from a rack is the
// number of letters it needs that the rack lacks, and the bitmask rules out
// most far words without counting.
interface WordIndex {
  words: string[];
  counts: Uint8Array[];
  masks: Uint32Array;
  score: Uint16Array;
}

const indexes = new WeakMap<string[], WordIndex>();

function indexFor(dictionary: string[]): WordIndex {
  let index = indexes.get(dictionary);
  if (index) return index;
  const n = dictionary.length;
  index = {
    words: dictionary,
    counts: new Array(n),
    masks: new Uint32Array(n),
    score: new Uint16Array(n),
  };
  for (let i = 0; i < n; i++) {
    const word = dictionary[i];
    const counts = new Uint8Array(26);
    let mask = 0;
    let base = 0;
    for (const c of word) {
      const k = code(c);
      counts[k]++;
      mask |= 1 << k;
      base += LETTER_VALUE[c] ?? 0;
    }
    index.counts[i] = counts;
    index.masks[i] = mask;
    index.score[i] = wordScore(base, word.length);
  }
  indexes.set(dictionary, index);
  return index;
}

// Letters word `i` needs beyond `rack`, or Infinity once that's over `limit`.
function missing(
  index: WordIndex,
  i: number,
  rack: Uint8Array,
  mask: number,
  limit: number,
): number {
  if (popcount(index.masks[i] & ~mask) > limit) return Infinity;
  const counts = index.counts[i];
  let short = 0;
  for (let k = 0; k < 26; k++) {
    if (counts[k] > rack[k]) {
      short += counts[k] - rack[k];
      if (short > limit) return Infinity;
    }
  }
  return short;
}

// Mean discounted score of the TOP_K best words within MAX_MISSING letters.
function potential(index: WordIndex, rack: Uint8Array): number {
  const top: number[] = [];
  const mask = rackMask(rack);
  for (let i = 0; i < index.words.length; i++) {
    const m = missing(index, i, rack, mask, MAX_MISSING);
    if (m === Infinity) continue;
    const value = index.score[i] * MISSING_DISCOUNT ** m;
    if (top.length < TOP_K) {
      top.push(value);
      top.sort((a, b) => b - a);
    } else if (value > top[TOP_K - 1]) {
      top[TOP_K - 1] = value;
      top.sort((a, b) => b - a);
    }
  }
  return top.reduce((a, b) => a + b, 0) / TOP_K;
}

function rackCounts(rack: Map<string, number>): Uint8Array {
  const counts = new Uint8Array(26);
  for (const [c, n] of rack) counts[code(c)] += n;
  return counts;
}

function rackMask(rack: Uint8Array): number {
  let mask = 0;
  for (let k = 0; k < 26; k++) if (rack[k] > 0) mask |= 1 << k;
  return mask;
}

function popcount(x: number): number {
  let n = 0;
  for (; x; x &= x - 1) n++;
  return n;
}

function code(c: string): number {
  return c.charCodeAt(0) - 65;
}

// Mirrors the module's reward: floor(base * length multiplier).
function wordScore(base: number, len: number): number {
  const mult = len <= 3 ? 1 : len === 4 ? 1.5 : len === 5 ? 2 : len === 6 ? 2.5 : 3;
  return Math.floor(base * mult);
}
