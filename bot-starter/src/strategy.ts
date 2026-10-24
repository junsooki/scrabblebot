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
  dictionary: string[]; // Sorted uppercase list, shared at startup
}

/**
 * Try to form a word using letters from `myRack`.
 * Return the word (uppercase) to play, or null to skip.
 *
 * Strategy (after Appel & Jacobson, "The World's Fastest Scrabble Program",
 * CACM 1988): store the lexicon in a trie and backtrack through it, only
 * following edges for letters still on the rack, so the search never visits
 * a prefix the rack can't spell. There's no board here, so the paper's
 * anchors and cross-checks drop out; like the paper, we play the
 * highest-scoring move found.
 *
 * Keep this fast. The bot runs it on every rack change, and while it runs
 * the bot can't see the match move on: a search that takes longer than an
 * auction means bids on closed auctions and words from spent letters.
 */
export function chooseWord(ctx: WordContext): string | null {
  const root = trieFor(ctx.dictionary);
  const rack = new Map(ctx.myRack);
  let best: string | null = null;
  let bestScore = 0;
  const path: string[] = [];

  const search = (node: TrieNode, base: number) => {
    if (node.word) {
      const score = wordScore(base, path.length);
      if (score > bestScore || (score === bestScore && best !== null && path.length > best.length)) {
        best = path.join("");
        bestScore = score;
      }
    }
    for (const [c, child] of node.children) {
      const n = rack.get(c) ?? 0;
      if (n === 0) continue;
      rack.set(c, n - 1);
      path.push(c);
      search(child, base + (LETTER_VALUE[c] ?? 0));
      path.pop();
      rack.set(c, n);
    }
  };
  search(root, 0);
  return best;
}

// Mirrors the module's reward: floor(base * length multiplier).
function wordScore(base: number, len: number): number {
  const mult = len <= 3 ? 1 : len === 4 ? 1.5 : len === 5 ? 2 : len === 6 ? 2.5 : 3;
  return Math.floor(base * mult);
}

interface TrieNode {
  word: boolean;
  children: Map<string, TrieNode>;
}

// The dictionary is shared for the bot's lifetime, so build its trie once.
const tries = new WeakMap<string[], TrieNode>();

function trieFor(dictionary: string[]): TrieNode {
  let root = tries.get(dictionary);
  if (root) return root;
  root = { word: false, children: new Map() };
  for (const word of dictionary) {
    let node = root;
    for (const c of word) {
      let next = node.children.get(c);
      if (!next) {
        next = { word: false, children: new Map() };
        node.children.set(c, next);
      }
      node = next;
    }
    node.word = true;
  }
  tries.set(dictionary, root);
  return root;
}
