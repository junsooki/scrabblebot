// ScrabbleBot bot starter.
//
// First run, with a token from the website:
//   On your team's page, click "Generate token", then run
//   `BOT_NAME=<name> BOT_TOKEN=<token> npm start`.
//
// Or with a nonce from the CLI:
//   1. `spacetime call scrabblebot mint_credential_nonce`
//      `spacetime sql 'SELECT code FROM my_nonces ORDER BY expires_at DESC'`
//   2. `BOT_NAME=<name> BOT_NONCE=<code> npm start`
//      The bot connects fresh and redeems the nonce.
//
// Either way the bot saves its token to .token-<BOT_NAME>, so later runs
// need only `BOT_NAME=<name> npm start`.
//
// Entering a tournament: `BOT_NAME=<name> TOURNAMENT_ID=<id> npm start`
// registers the bot for that tournament when it connects. Only the bot
// itself can register, so this has to come from the bot's process.
//
// Edit ./src/strategy.ts to customise how your bot bids and plays words.

import {
  DbConnection,
  type EventContext,
  type ErrorContext,
} from "./module_bindings/index.js";
import { Identity } from "spacetimedb";
import * as fs from "node:fs";
import * as path from "node:path";
import { MatchBrain, indexFor } from "./strategy.js";

const HOST = process.env.STDB_HOST ?? "https://maincloud.spacetimedb.com";
const DB_NAME = process.env.STDB_DB ?? "scrabblebot";
const BOT_NAME = process.env.BOT_NAME ?? "bot";
const BOT_TOKEN = process.env.BOT_TOKEN; // from the website, first run only
const BOT_NONCE = process.env.BOT_NONCE; // from the CLI, first run only
const TOURNAMENT_ID = process.env.TOURNAMENT_ID;
const TOKEN_PATH = path.join(process.cwd(), `.token-${BOT_NAME}`);

function loadToken(): string | undefined {
  try {
    return fs.readFileSync(TOKEN_PATH, "utf8").trim() || undefined;
  } catch {
    return undefined;
  }
}
function saveToken(tok: string) {
  fs.writeFileSync(TOKEN_PATH, tok);
}

// Load the shared wordlist so the bot can pick playable words locally.
const dictionaryPath = path.join(
  path.dirname(new URL(import.meta.url).pathname),
  "..",
  "..",
  "spacetimedb",
  "wordlist.txt",
);
const DICTIONARY: string[] = fs.existsSync(dictionaryPath)
  ? fs
      .readFileSync(dictionaryPath, "utf8")
      .split("\n")
      .map((l) => l.trim().toUpperCase())
      .filter((l) => l.length >= 2)
  : [];
// Build the word index now rather than on the first auction.
indexFor(DICTIONARY);

// Don't bother bidding on an auction closing sooner than this: the bid
// would arrive late (one-way latency is about 40 ms).
const MIN_TIME_LEFT_MS = 60;

let myIdentity: Identity | null = null;
let myBotId: bigint | null = null;
let subscribedAsBot = false;
const bids = new Map<string, number>(); // auction id -> amount we bid
const brains = new Map<string, MatchBrain>();
const playing = new Set<string>(); // matches with a word in flight

function resolveMyBotId(conn: DbConnection): bigint | null {
  if (!myIdentity) return null;
  const cred = conn.db.botCredential.identity.find(myIdentity);
  return cred ? cred.botId : null;
}

function rackForMatch(
  conn: DbConnection,
  matchId: bigint,
): Map<string, number> {
  const rack = new Map<string, number>();
  for (const h of conn.db.myRack.iter()) {
    if (h.matchId !== matchId) continue;
    rack.set(h.letter, (rack.get(h.letter) ?? 0) + h.count);
  }
  return rack;
}

function participantForMatch(
  conn: DbConnection,
  matchId: bigint,
): { balance: number; score: number } | null {
  if (myBotId === null) return null;
  for (const p of conn.db.matchParticipant.iter()) {
    if (p.matchId !== matchId) continue;
    if (p.botId !== myBotId) continue;
    return { balance: Number(p.balance), score: Number(p.score) };
  }
  return null;
}

function isRunning(conn: DbConnection, matchId: bigint): boolean {
  return conn.db.matchState.id.find(matchId)?.status.tag === "Running";
}

function lettersLeft(conn: DbConnection, matchId: bigint): number {
  return conn.db.matchState.id.find(matchId)?.bagTotal ?? 0;
}

// The brain for a match we're playing, created on first use.
function brainFor(conn: DbConnection, matchId: bigint): MatchBrain | null {
  if (myBotId === null || !participantForMatch(conn, matchId)) return null;
  const key = String(matchId);
  let brain = brains.get(key);
  if (!brain) {
    let players = 0;
    for (const p of conn.db.matchParticipant.iter()) {
      if (p.matchId === matchId) players++;
    }
    brain = new MatchBrain(DICTIONARY, players, String(myBotId));
    brains.set(key, brain);
  }
  return brain;
}

// A reducer call fails when the module refuses it, often for an expected
// reason: a word played just after its match ended, a bid that missed the
// auction's window. Log it and keep playing; an uncaught failure would stop
// the bot.
function logFailure(what: string) {
  return (err: unknown) =>
    console.warn(
      `[${BOT_NAME}] ${what} failed: ${err instanceof Error ? err.message : err}`,
    );
}

// Bid on an open auction, or raise our bid if we can now afford more (say,
// after a word paid out mid-auction).
function tryBid(
  conn: DbConnection,
  auctionId: bigint,
  matchId: bigint,
  letter: string,
  closesAtMicros: bigint,
) {
  const brain = brainFor(conn, matchId);
  const participant = participantForMatch(conn, matchId);
  if (!brain || !participant) return;
  if (Number(closesAtMicros / 1000n) - Date.now() < MIN_TIME_LEFT_MS) return;
  const key = String(auctionId);
  const amount = brain.bid(
    letter,
    rackForMatch(conn, matchId),
    participant.balance,
    lettersLeft(conn, matchId),
  );
  if (amount <= (bids.get(key) ?? 0)) return;
  bids.set(key, amount);
  conn.reducers
    .submitBid({ auctionId, amount: BigInt(amount) })
    .catch(logFailure(`bid on auction ${auctionId}`));
  console.log(
    `[${BOT_NAME}] bid ${amount} on '${letter}' (match ${matchId}, auction ${auctionId}, alpha ${brain.alpha.toFixed(2)})`,
  );
}

// Re-bid on the match's open auction, if any.
function rebid(conn: DbConnection, matchId: bigint) {
  for (const a of conn.db.auction.iter()) {
    if (a.matchId === matchId && a.status.tag === "Open") {
      tryBid(conn, a.id, a.matchId, a.letter, a.closesAt.microsSinceUnixEpoch);
    }
  }
}

// Play a word if the brain wants to. One word in flight per match; when it
// lands, look again, so the endgame can dump several words in one auction.
function tryPlayWord(conn: DbConnection, matchId: bigint) {
  const key = String(matchId);
  if (playing.has(key) || !isRunning(conn, matchId)) return;
  const brain = brainFor(conn, matchId);
  const participant = participantForMatch(conn, matchId);
  if (!brain || !participant) return;
  const word = brain.word(
    rackForMatch(conn, matchId),
    participant.balance,
    lettersLeft(conn, matchId),
  );
  if (!word) return;
  console.log(`[${BOT_NAME}] match ${matchId}: playing '${word}'`);
  playing.add(key);
  conn.reducers
    .submitWord({ matchId, word })
    .catch(logFailure(`playing '${word}'`))
    .finally(() => {
      playing.delete(key);
      tryPlayWord(conn, matchId);
      rebid(conn, matchId);
    });
}

function onConnect(conn: DbConnection, identity: Identity, token: string) {
  myIdentity = identity;
  saveToken(token);
  console.log(`[${BOT_NAME}] connected as ${identity.toHexString()}`);

  // Subscribe only to what the bot needs. Phase 1 finds our bot id; phase 2
  // is scoped to it. (Subscribing to all tables pulls the server's whole
  // history, ~1M rows, and stalls the bot for seconds.)
  conn
    .subscriptionBuilder()
    .onApplied(() => {
      // Resolve our bot id, possibly after claiming a nonce.
      myBotId = resolveMyBotId(conn);
      if (myBotId !== null) return subscribeAsBot(conn);
      if (BOT_NONCE) {
        console.log(`[${BOT_NAME}] claiming credential with nonce…`);
        conn.reducers
          .claimCredential({ code: BOT_NONCE })
          .catch(logFailure("claiming the credential"));
        // botCredential.onInsert picks it up; give up if it doesn't come.
        setTimeout(() => {
          if (myBotId === null) {
            console.error(
              `[${BOT_NAME}] couldn't claim credential. Bad / expired nonce?`,
            );
            process.exit(1);
          }
        }, 5000);
      } else {
        console.error(
          `[${BOT_NAME}] no BotCredential for this token. Set BOT_TOKEN or BOT_NONCE and re-run.`,
        );
        process.exit(1);
      }
    })
    .subscribe(
      `SELECT * FROM bot_credential WHERE identity = 0x${identity.toHexString()}`,
    );

  const replay = (ctx: EventContext) => ctx.event.tag === "SubscribeApplied";

  conn.db.botCredential.onInsert(() => {
    if (myBotId !== null) return;
    myBotId = resolveMyBotId(conn);
    if (myBotId !== null) subscribeAsBot(conn);
  });
  conn.db.auction.onInsert((ctx: EventContext, a) => {
    brainFor(conn, a.matchId)?.noteAuction(String(a.id), a.letter);
    // Replayed rows are handled by bootstrapActivity once they're all in.
    if (replay(ctx) || a.status.tag !== "Open") return;
    tryBid(conn, a.id, a.matchId, a.letter, a.closesAt.microsSinceUnixEpoch);
    // The bag shrinks every auction, which can turn a held rack into a
    // play, so look again even if the rack didn't change.
    tryPlayWord(conn, a.matchId);
  });
  conn.db.auctionResult.onInsert((ctx: EventContext, r) => {
    const brain = brainFor(conn, r.matchId);
    if (!brain) return;
    const winner = r.winnerBotId == null ? null : String(r.winnerBotId);
    brain.noteResult(
      String(r.auctionId),
      r.letter,
      winner,
      Number(r.paid),
      participantForMatch(conn, r.matchId)?.balance ?? 0,
      lettersLeft(conn, r.matchId),
    );
    if (replay(ctx)) return;
    console.log(
      `[${BOT_NAME}] match ${r.matchId} '${r.letter}' → ${winner === String(myBotId) ? "us" : `bot ${winner ?? "none"}`} paid ${r.paid}`,
    );
  });
  conn.db.wordPlay.onInsert((_ctx, w) => {
    brainFor(conn, w.matchId)?.noteWord(String(w.id), String(w.botId), w.word);
  });
  conn.db.myRack.onInsert((ctx, h) => {
    if (!replay(ctx)) tryPlayWord(conn, h.matchId);
  });
  conn.db.myRack.onUpdate((_ctx, _old, h) => tryPlayWord(conn, h.matchId));
  conn.db.matchParticipant.onUpdate((_ctx, old, neu) => {
    // A word paid out: we may afford a higher bid on the open auction.
    if (neu.botId === myBotId && neu.balance > old.balance) rebid(conn, neu.matchId);
  });

  // When a match this bot was in ends, hop back into the lobby.
  conn.db.matchState.onUpdate((_ctx, old, neu) => {
    if (myBotId === null) return;
    if (old.status.tag !== "Ended" && neu.status.tag === "Ended") {
      brains.delete(String(neu.id));
      if (participantForMatch(conn, neu.id)) {
        console.log(`[${BOT_NAME}] match ${neu.id} ended; rejoining lobby`);
        joinLobby(conn);
      }
    }
  });
}

function subscribeAsBot(conn: DbConnection) {
  if (subscribedAsBot || myBotId === null) return;
  subscribedAsBot = true;
  const me = myBotId;
  // Recent enough to cover a whole running match (~70 s), so a restart
  // mid-match can rebuild its bookkeeping.
  const since = new Date(Date.now() - 120_000).toISOString();
  const recent = new Date(Date.now() - 5_000).toISOString();
  const ours = `JOIN match_participant mp ON t.match_id = mp.match_id WHERE mp.bot_id = ${me}`;
  conn
    .subscriptionBuilder()
    .onApplied(() => {
      const bot = conn.db.bot.id.find(me);
      console.log(
        `[${BOT_NAME}] acting as bot '${bot?.name ?? "?"}' (id ${me})`,
      );
      bootstrapActivity(conn);
    })
    .subscribe([
      `SELECT * FROM my_rack`,
      `SELECT t.* FROM match_participant t ${ours}`,
      `SELECT t.* FROM match_state t JOIN match_participant mp ON t.id = mp.match_id WHERE mp.bot_id = ${me}`,
      `SELECT t.* FROM auction t ${ours} AND t.closes_at > '${recent}'`,
      `SELECT t.* FROM auction_result t ${ours} AND t.closed_at > '${since}'`,
      `SELECT t.* FROM word_play t ${ours} AND t.played_at > '${since}'`,
      `SELECT l.* FROM lobby l JOIN lobby_member lm ON l.id = lm.lobby_id WHERE lm.bot_id = ${me}`,
      `SELECT * FROM lobby_member WHERE bot_id = ${me}`,
      `SELECT * FROM bot WHERE id = ${me}`,
    ]);
}

function joinLobby(conn: DbConnection) {
  if (myBotId === null) return;
  const inRunning = Array.from(conn.db.matchParticipant.iter()).some(
    (p) => p.botId === myBotId && isRunning(conn, p.matchId),
  );
  if (inRunning) return;
  const openLobby = Array.from(conn.db.lobby.iter()).find(
    (l) => l.status.tag === "Open",
  );
  const alreadyIn =
    openLobby !== undefined &&
    Array.from(conn.db.lobbyMember.iter()).some(
      (lm) => lm.lobbyId === openLobby.id && lm.botId === myBotId,
    );
  if (alreadyIn) return;
  console.log(`[${BOT_NAME}] joining lobby`);
  conn.reducers.joinLobby({}).catch(logFailure("joining the lobby"));
}

function bootstrapActivity(conn: DbConnection) {
  if (TOURNAMENT_ID) {
    conn.reducers
      .registerForTournament({ tournamentId: BigInt(TOURNAMENT_ID) })
      .then(() =>
        console.log(`[${BOT_NAME}] registered for tournament ${TOURNAMENT_ID}`),
      )
      .catch(logFailure(`registering for tournament ${TOURNAMENT_ID}`));
  }
  joinLobby(conn);
  for (const a of conn.db.auction.iter()) {
    if (a.status.tag === "Open") {
      tryBid(conn, a.id, a.matchId, a.letter, a.closesAt.microsSinceUnixEpoch);
    }
  }
  for (const p of conn.db.matchParticipant.iter()) {
    if (p.botId === myBotId) tryPlayWord(conn, p.matchId);
  }
}

function main() {
  console.log(`[${BOT_NAME}] connecting to ${DB_NAME} at ${HOST}`);
  DbConnection.builder()
    .withUri(HOST)
    .withDatabaseName(DB_NAME)
    .withToken(BOT_TOKEN ?? loadToken())
    .withAutomaticReconnect()
    .onConnect(onConnect)
    // The SDK replays our subscriptions after a reconnect.
    .onAutomaticReconnect(() => console.log(`[${BOT_NAME}] reconnected`))
    .onConnectError((_ctx: ErrorContext, err: Error, nextAttempt?: number) => {
      console.error("connect error:", err.message);
      // The first connection isn't retried; exit so a supervisor restarts us.
      if (nextAttempt === undefined) process.exit(1);
    })
    .onDisconnect(() => console.log(`[${BOT_NAME}] disconnected`))
    .build();
}

main();
