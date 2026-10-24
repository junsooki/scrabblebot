// Edit this file to implement your bidding and word-forming strategy.
//
// One MatchBrain per match. It keeps exact bag bookkeeping (what's still
// unseen), values the rack by the target words it can reach, bids that value
// converted to coins at a paced exchange rate, and decides when to cash in a
// word by comparing equity (score now + value of what's left) against
// holding. Background and derivations: Appel & Jacobson (1988) for the
// lexicon search idea, Maven/Quackle "equity" for play timing, and budget
// pacing from the repeated-auction literature (Balseiro & Gur).

// Standard Scrabble letter values — useful for valuation.
export const LETTER_VALUE: Record<string, number> = {
  A: 1, B: 3, C: 3, D: 2, E: 1, F: 4, G: 2, H: 4, I: 1, J: 8,
  K: 5, L: 1, M: 3, N: 1, O: 1, P: 3, Q: 10, R: 1, S: 1, T: 1,
  U: 1, V: 4, W: 4, X: 8, Y: 4, Z: 10,
};

// The module's bag: standard Scrabble distribution, no blanks, 98 tiles.
const FULL_BAG: Record<string, number> = {
  A: 9, B: 2, C: 2, D: 4, E: 12, F: 2, G: 3, H: 2, I: 9, J: 1, K: 1, L: 4, M: 2,
  N: 6, O: 8, P: 2, Q: 1, R: 6, S: 4, T: 6, U: 4, V: 2, W: 2, X: 1, Y: 2, Z: 1,
};

// Tunable parameters. See research notes for where the defaults come from;
// tune them with `npm run sim`.
export interface Theta {
  alpha0: number;      // starting coins paid per point of value
  alphaMin: number;
  alphaMax: number;
  eta: number;         // pacing step
  rho0: number;        // extra value of a reward that comes back as money to spend
  omega: [number, number, number]; // our chance to win a given copy, by tier
  pi: [number, number, number];    // expected price when we win, by tier
  gamma: number;       // weight of the second-best disjoint word
  maxMissing: number;  // furthest target words considered
  topN: number;        // target words kept per rack
  kEnd: number;        // auctions left when endgame escalation starts
  kappa: number;       // endgame escalation strength
  lottery: number;     // floor bid on every letter early on
  playK: number;       // playable words compared when deciding to play
  epsilon: number;     // play when equity is within this of holding
  minPlayLen: number;  // shortest word worth playing before the endgame
  floorK: number;      // floor bid as a share of the letter's market price
}

export const DEFAULT_THETA: Theta = {
  alpha0: 1.7, alphaMin: 1, alphaMax: 4, eta: 0.08, rho0: 0.5,
  omega: [0.45, 0.4, 0.35], pi: [12, 19, 32], gamma: 0.7, maxMissing: 2,
  topN: 64, kEnd: 8, kappa: 1.5, lottery: 2, playK: 6, epsilon: 1,
  minPlayLen: 6, floorK: 0.9,
};

const STARTING_RACK = 5;

export class MatchBrain {
  alpha: number;
  private readonly index: WordIndex;
  private readonly auctioned = new Map<string, number>(); // auction id -> letter code
  private readonly results = new Set<string>();
  private readonly words = new Set<string>();
  private readonly won = new Map<string, Uint8Array>();    // bot id -> letters won
  private readonly played = new Map<string, Uint8Array>(); // bot id -> letters played
  private readonly prices: number[];
  private income = 0; // money our current plans are expected to bring back
  private rack = new Uint8Array(26); // our actual rack at the last decision
  private planCache = new Map<string, Plan[]>();
  private planCacheLeft = -1;

  constructor(
    dictionary: string[],
    readonly players: number,
    readonly me: string,
    readonly theta: Theta = DEFAULT_THETA,
  ) {
    this.index = indexFor(dictionary);
    this.alpha = theta.alpha0;
    this.prices = [...theta.pi];
  }

  // Bookkeeping. Each is idempotent, so replayed rows are harmless.

  noteAuction(auctionId: string, letter: string) {
    this.auctioned.set(auctionId, code(letter));
  }

  noteResult(
    auctionId: string,
    letter: string,
    winner: string | null,
    paid: number,
    myBalance: number,
    lettersLeft: number,
  ) {
    this.noteAuction(auctionId, letter);
    if (this.results.has(auctionId)) return;
    this.results.add(auctionId);
    if (winner === null) return;
    counts(this.won, winner)[code(letter)]++;
    const t = tier(code(letter));
    this.prices[t] = 0.7 * this.theta.pi[t] + 0.3 * (0.8 * this.prices[t] + 0.2 * paid);
    // Pacing: nudge the exchange rate toward spending our money (plus what
    // our plans should bring back) evenly over the auctions left.
    const spent = winner === this.me ? paid : 0;
    const target = (myBalance + spent + this.income) / Math.max(1, lettersLeft + 1);
    const { eta, alphaMin, alphaMax } = this.theta;
    this.alpha = clamp(
      this.alpha * Math.exp((eta * (target - spent)) / Math.max(target, 1)),
      alphaMin,
      alphaMax,
    );
  }

  noteWord(wordId: string, botId: string, word: string) {
    if (this.words.has(wordId)) return;
    this.words.add(wordId);
    const c = counts(this.played, botId);
    for (const ch of word) c[code(ch)]++;
  }

  // Decisions.

  bid(letter: string, rack: Map<string, number>, balance: number, lettersLeft: number): number {
    // The final tile arrives as the match ends and can never be played.
    if (lettersLeft <= 0 || balance <= 0) return 0;
    const r = rackCounts(rack);
    this.setRack(r);
    const before = this.value(r, lettersLeft, true);
    r[code(letter)]++;
    const gain = this.value(r, lettersLeft, false) - before;
    const { kEnd, kappa, lottery } = this.theta;
    let a = this.alpha;
    if (lettersLeft <= kEnd) a *= 1 + (kappa * (kEnd - lettersLeft)) / kEnd;
    // Never bid far below the market (about 9 + 2.4 × letter value): an
    // empty-ish rack sees little gain in any one letter but still needs tiles.
    const market = (9 + 2.4 * LETTER_VALUE[letter]) * (a / this.theta.alpha0);
    let bid = Math.round(Math.max(gain * a, this.theta.floorK * market));
    // Money is worthless once the bag is empty: late, spend it on anything useful.
    if (lettersLeft <= kEnd && gain > 0) bid = Math.max(bid, Math.floor(balance / lettersLeft));
    if (lettersLeft > 15) bid = Math.max(bid, lottery);
    return Math.min(bid, balance);
  }

  word(rack: Map<string, number>, balance: number, lettersLeft: number): string | null {
    const r = rackCounts(rack);
    this.setRack(r);
    const endgame = lettersLeft <= this.theta.kEnd;
    // Before the endgame, short words just burn tiles a long word needs.
    const playable = this.playable(r).filter(
      (i) => endgame || this.index.lens[i] >= this.theta.minPlayLen,
    );
    if (playable.length === 0) return null;
    // Last chance: the tile won now is only playable while the final auction
    // runs, and nothing is playable after. Cash in everything, best first.
    if (lettersLeft <= 1) return this.index.words[playable[0]];
    const rho = this.rho(lettersLeft);
    const hold = this.value(r, lettersLeft, true);
    const lowCash = balance < this.prices[0] ? (this.prices[0] - balance) / this.alpha : 0;
    let best = -1;
    let bestEquity = -Infinity;
    for (const i of playable) {
      const rest = r.slice();
      const off = i * 26;
      for (let k = 0; k < 26; k++) rest[k] -= this.index.counts[off + k];
      const equity = this.index.score[i] * (1 + rho) + this.value(rest, lettersLeft, false) + lowCash;
      if (equity > bestEquity) {
        bestEquity = equity;
        best = i;
      }
    }
    return bestEquity >= hold - this.theta.epsilon ? this.index.words[best] : null;
  }

  // Valuation.

  private setRack(r: Uint8Array) {
    if (r.join() !== this.rack.join()) this.planCache.clear();
    this.rack = r.slice();
  }

  private rho(lettersLeft: number): number {
    return this.theta.rho0 * clamp((lettersLeft - 4) / 40, 0, 1);
  }

  // Unseen copies of each letter that we could still win and use.
  private usable(lettersLeft: number): Float64Array {
    const unseen = new Float64Array(26);
    const auctioned = new Uint8Array(26);
    for (const c of this.auctioned.values()) auctioned[c]++;
    const none = new Uint8Array(26);
    // Letters known to have come from starting racks: all of ours (what we
    // hold or played beyond what we won) and opponents' as they play them.
    const reveal = new Uint8Array(26);
    const myWon = this.won.get(this.me) ?? none;
    const myPlayed = this.played.get(this.me) ?? none;
    for (let k = 0; k < 26; k++) {
      reveal[k] = Math.max(0, this.rack[k] + myPlayed[k] - myWon[k]);
    }
    let revealed = 0;
    for (const [bot, played] of this.played) {
      if (bot === this.me) continue;
      const won = this.won.get(bot) ?? none;
      for (let k = 0; k < 26; k++) {
        const fromStart = Math.max(0, played[k] - won[k]);
        revealed += fromStart;
        reveal[k] += fromStart;
      }
    }
    for (let k = 0; k < 26; k++) {
      unseen[k] = Math.max(0, FULL_BAG[String.fromCharCode(65 + k)] - auctioned[k] - reveal[k]);
    }
    // Opponents' starting tiles we haven't seen yet are also unseen but not
    // in the bag, so only a share of the unseen copies will be auctioned.
    const hidden = Math.max(0, (this.players - 1) * STARTING_RACK - revealed);
    const share = Math.max(0, lettersLeft - 1) / Math.max(1, lettersLeft + hidden);
    for (let k = 0; k < 26; k++) unseen[k] *= share;
    return unseen;
  }

  // V(rack): best target word plus a discounted second word that fits in
  // what's left. With `track`, remember the money it should bring back.
  private value(rack: Uint8Array, lettersLeft: number, track: boolean): number {
    const plans = this.plans(rack, lettersLeft);
    const { gamma } = this.theta;
    let best = 0;
    let income = 0;
    for (let a = 0; a < plans.length; a++) {
      const pa = plans[a];
      let second: Plan | null = null;
      for (let b = 0; b < plans.length; b++) {
        if (b === a) continue;
        if (fits(pa.used, plans[b].used, rack)) {
          second = plans[b];
          break; // plans are sorted best first
        }
      }
      const v = pa.ev + (second ? gamma * second.ev : 0);
      if (v > best) {
        best = v;
        income = pa.p * pa.s + (second ? gamma * second.p * second.s : 0);
      }
    }
    if (track) this.income = income;
    return best;
  }

  // The topN target words within maxMissing letters of `rack`, by expected
  // value in points: P(we win the missing letters) × score, minus what
  // they'd cost converted to points.
  private plans(rack: Uint8Array, lettersLeft: number): Plan[] {
    if (lettersLeft !== this.planCacheLeft) {
      this.planCache.clear();
      this.planCacheLeft = lettersLeft;
    }
    const key = rack.join(",");
    const hit = this.planCache.get(key);
    if (hit) return hit;

    const { maxMissing, topN, omega } = this.theta;
    const idx = this.index;
    const usable = this.usable(lettersLeft);
    const rho = this.rho(lettersLeft);
    let total = 0;
    for (let k = 0; k < 26; k++) total += rack[k];
    const mask = maskOf(rack);
    const top: Plan[] = [];
    const n = idx.words.length;
    for (let i = idx.startAtLen(total + maxMissing); i < n; i++) {
      if (popcount(idx.masks[i] & ~mask) > maxMissing) continue;
      const off = i * 26;
      let missing = 0;
      for (let k = 0; k < 26; k++) {
        const short = idx.counts[off + k] - rack[k];
        if (short > 0) missing += short;
      }
      if (missing > maxMissing) continue;
      const s = idx.score[i];
      let p = 1;
      let cost = 0;
      if (missing > 0) {
        for (let k = 0; k < 26; k++) {
          const short = idx.counts[off + k] - rack[k];
          if (short <= 0) continue;
          const t = tier(k);
          p *= poissonTail(usable[k] * omega[t], short);
          cost += short * this.prices[t];
        }
      }
      const ev = missing === 0 ? s * (1 + rho) : p * s * (1 + rho / 2) - cost / this.alpha;
      if (ev <= 0) continue;
      if (top.length === topN && ev <= top[topN - 1].ev) continue;
      const used = new Uint8Array(26);
      for (let k = 0; k < 26; k++) used[k] = Math.min(idx.counts[off + k], rack[k]);
      insertSorted(top, { i, ev, p, s, used }, topN);
    }
    this.planCache.set(key, top);
    return top;
  }

  // Up to playK playable words, best score first.
  private playable(rack: Uint8Array): number[] {
    const idx = this.index;
    let total = 0;
    for (let k = 0; k < 26; k++) total += rack[k];
    const mask = maskOf(rack);
    const found: number[] = [];
    const n = idx.words.length;
    for (let i = idx.startAtLen(total); i < n; i++) {
      if (idx.masks[i] & ~mask) continue;
      const off = i * 26;
      let ok = true;
      for (let k = 0; k < 26; k++) {
        if (idx.counts[off + k] > rack[k]) {
          ok = false;
          break;
        }
      }
      if (ok) found.push(i);
    }
    found.sort((a, b) => idx.score[b] - idx.score[a] || idx.lens[b] - idx.lens[a]);
    return found.slice(0, this.theta.playK);
  }
}

interface Plan {
  i: number;        // word index
  ev: number;       // expected value in points
  p: number;        // chance of completing it
  s: number;        // its score
  used: Uint8Array; // rack letters it uses
}

// Whether two plans' rack letters fit in the rack together.
function fits(a: Uint8Array, b: Uint8Array, rack: Uint8Array): boolean {
  for (let k = 0; k < 26; k++) if (a[k] + b[k] > rack[k]) return false;
  return true;
}

function insertSorted(top: Plan[], plan: Plan, limit: number) {
  let j = top.length;
  while (j > 0 && top[j - 1].ev < plan.ev) j--;
  top.splice(j, 0, plan);
  if (top.length > limit) top.pop();
}

// P(X >= k) for X ~ Poisson(mu).
function poissonTail(mu: number, k: number): number {
  if (mu <= 0) return 0;
  let term = Math.exp(-mu);
  let below = 0;
  for (let j = 0; j < k; j++) {
    below += term;
    term *= mu / (j + 1);
  }
  return Math.max(0, 1 - below);
}

// 0: 1-2 point letters, 1: B C F H M P V W Y, 2: J K Q X Z.
function tier(k: number): 0 | 1 | 2 {
  const v = LETTER_VALUE[String.fromCharCode(65 + k)];
  return v <= 2 ? 0 : v >= 5 ? 2 : 1;
}

// The dictionary as letter signatures: per word its letter counts (flat,
// 26 per word) and a bitmask of the letters it uses, sorted longest first so
// a scan can skip words too long to reach.
interface WordIndex {
  words: string[];
  counts: Uint8Array;
  masks: Uint32Array;
  score: Uint16Array;
  lens: Uint8Array;
  startAtLen(len: number): number; // first word no longer than len
}

const indexes = new WeakMap<string[], WordIndex>();

export function indexFor(dictionary: string[]): WordIndex {
  const cached = indexes.get(dictionary);
  if (cached) return cached;
  const words = [...dictionary].sort((a, b) => b.length - a.length);
  const n = words.length;
  const counts = new Uint8Array(n * 26);
  const masks = new Uint32Array(n);
  const score = new Uint16Array(n);
  const lens = new Uint8Array(n);
  for (let i = 0; i < n; i++) {
    const word = words[i];
    let base = 0;
    for (const ch of word) {
      const k = code(ch);
      counts[i * 26 + k]++;
      masks[i] |= 1 << k;
      base += LETTER_VALUE[ch] ?? 0;
    }
    lens[i] = word.length;
    score[i] = wordScore(base, word.length);
  }
  const maxLen = n ? lens[0] : 0;
  const start = new Int32Array(maxLen + 2);
  for (let len = 0, i = n; len <= maxLen + 1; len++) {
    while (i > 0 && lens[i - 1] <= len) i--;
    start[len] = i;
  }
  const index: WordIndex = {
    words, counts, masks, score, lens,
    startAtLen: (len) => (len > maxLen ? 0 : start[Math.max(0, len)]),
  };
  indexes.set(dictionary, index);
  return index;
}

function counts(map: Map<string, Uint8Array>, key: string): Uint8Array {
  let c = map.get(key);
  if (!c) map.set(key, (c = new Uint8Array(26)));
  return c;
}

function rackCounts(rack: Map<string, number>): Uint8Array {
  const c = new Uint8Array(26);
  for (const [ch, n] of rack) c[code(ch)] += n;
  return c;
}

function maskOf(rack: Uint8Array): number {
  let mask = 0;
  for (let k = 0; k < 26; k++) if (rack[k] > 0) mask |= 1 << k;
  return mask;
}

function popcount(x: number): number {
  let n = 0;
  for (; x; x &= x - 1) n++;
  return n;
}

function clamp(x: number, lo: number, hi: number): number {
  return Math.min(hi, Math.max(lo, x));
}

function code(c: string): number {
  return c.charCodeAt(0) - 65;
}

// Mirrors the module's reward: floor(base * length multiplier).
export function wordScore(base: number, len: number): number {
  const mult = len <= 3 ? 1 : len === 4 ? 1.5 : len === 5 ? 2 : len === 6 ? 2.5 : 3;
  return Math.floor(base * mult);
}
