"""
agent.py - YOUR poker bot. This one file is a complete web server.

Run it:
    python agent.py                  # listens on port 5008
    PORT=5010 python agent.py        # or pick a port

Play a match against a baseline (three windows):
    python agent.py                                 # your bot on port 5008
    python baseline-bots/baseline_random.py         # opponent on port 5009
    python judge_engine.py                          # judge: both halves (uses those two ports)

Check that the tournament can run it:
    python -m pytest -q              # tests in test_agent.py
    ./tools/check_build.sh           # builds and smoke-tests it like the tournament

TODO:
    1. Assign your agent a unique PARTICIPANT and AGENT_NAME below.
    2. Implement the three functions between "YOUR CODE HERE" and "END OF YOUR CODE".

Everything under "SERVER PLUMBING" talks to the judge. You normally never
need to change it, and if you break it the judge can't talk to your bot.
"""

import os
import traceback

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from evaluator import card_rank, card_suit, cards_to_str, omaha_score, warmup  # handy helpers, see evaluator.py
from game import Action, AgentConfig, HandResult, Observation, Street

# Build omaha_score's lookup table now, while the server starts, instead of
# during your first decision (where it would eat into your time bank).
# Do any other slow setup (loading a model, ...) here too: the server must
# answer within 15 seconds of starting.
warmup()

# rename as your team/bot's name
PARTICIPANT = "Straw-hat Crew"
AGENT_NAME = "Chopper"


# ===== YOUR CODE HERE =====
# ---------------------------------------------------------------------------
# Chopper - heads-up short-deck PLO5.
#
# How it thinks, in one paragraph:
#   1. Estimate "equity": how often we win the pot versus one random opponent
#      hand.  Preflop that comes from a calibrated hand-feature model; from the
#      flop on it comes from Monte Carlo simulation of opponent hands and the
#      missing board cards, scored with the official omaha_score table.
#   2. Compare that equity with the price of continuing (pot odds) and bet
#      sizes, using thresholds that shift with the opponent's measured
#      aggression and how often they fold.
#   3. Stay inside the 8 s time bank with a per-decision cap derived from how
#      much bank is left and how many hands remain.
#
# Every piece of per-half state lives in `memory`, never in globals.
# ---------------------------------------------------------------------------

import time as _time
import random as _random

# numpy is preinstalled in the tournament image.  If it is ever missing we fall
# back to plain-Python equity with fewer samples (still legal, just coarser).
try:
    import numpy as _np
except Exception:  # pragma: no cover
    _np = None

from evaluator import get_table as _get_table
from evaluator import omaha_score as _omaha_score

# --- tuning knobs ----------------------------------------------------------
POOL_SIZE = 512            # pre-drawn shuffles kept per remaining-deck size
BANK_SHARE = 0.55          # never spend more than this share of the time bank
SEC_PER_TRIAL = 0.0000135  # measured cost of one simulated showdown (both hands)
MIN_TRIALS, MAX_TRIALS = 10, 160

_PAIR_IDX = ((0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4))
_TRIPLE_IDX = ((0, 1, 2), (0, 1, 3), (0, 1, 4), (0, 2, 3), (0, 2, 4), (0, 3, 4),
               (1, 2, 3), (1, 2, 4), (1, 3, 4), (2, 3, 4))

# The evaluator stores its scores in a dict; sorted numpy arrays turn the 100
# lookups of one showdown into one vectorised searchsorted call.
_FAST = {"mask": None, "score": None, "pools": {}, "ready": False,
         "pa": None, "pb": None, "ta": None, "tb": None, "tc": None}


def _init_fast() -> None:
    """Build the vectorised hand-score table and the shuffle pool (once)."""
    if _np is None:
        return
    table = _get_table()
    n = len(table)
    masks = _np.fromiter(table.keys(), dtype=_np.int64, count=n)
    scores = _np.fromiter(table.values(), dtype=_np.int64, count=n)
    order = _np.argsort(masks, kind="stable")
    _FAST["mask"] = masks[order]
    _FAST["score"] = scores[order]
    _FAST["pa"] = _np.array([p[0] for p in _PAIR_IDX], dtype=_np.intp)
    _FAST["pb"] = _np.array([p[1] for p in _PAIR_IDX], dtype=_np.intp)
    _FAST["ta"] = _np.array([t[0] for t in _TRIPLE_IDX], dtype=_np.intp)
    _FAST["tb"] = _np.array([t[1] for t in _TRIPLE_IDX], dtype=_np.intp)
    _FAST["tc"] = _np.array([t[2] for t in _TRIPLE_IDX], dtype=_np.intp)
    # A fixed bank of random orderings per deck size: sampling then costs a
    # couple of gathers instead of a fresh shuffle on every decision.
    rng = _np.random.default_rng(20261107)
    pools = {}
    for size in range(20, 33):
        base = _np.tile(_np.arange(size, dtype=_np.int64), (POOL_SIZE, 1))
        pools[size] = rng.permuted(base, axis=1)
    _FAST["pools"] = pools
    _FAST["ready"] = True


_init_fast()


def _paired_scores(hole, opps, boards):
    """Best omaha_score for `hole` and for each row of `opps`, on `boards`.

    Returns (mine, theirs); `mine` is the same value repeated per row.
    """
    np = _np
    hb = np.int64(1) << np.asarray(hole, dtype=np.int64)
    hp = hb[_FAST["pa"]] | hb[_FAST["pb"]]                       # 10 hole pairs
    bb = np.int64(1) << boards
    bt = bb[:, _FAST["ta"]] | bb[:, _FAST["tb"]] | bb[:, _FAST["tc"]]  # 10 triples
    mine = (hp[None, :, None] | bt[:, None, :]).reshape(-1, 100)
    ob = np.int64(1) << opps
    hp_opp = ob[:, _FAST["pa"]] | ob[:, _FAST["pb"]]
    theirs = (hp_opp[:, :, None] | bt[:, None, :]).reshape(-1, 100)
    masks = np.concatenate((mine, theirs), axis=0)
    idx = np.searchsorted(_FAST["mask"], masks)
    np.clip(idx, 0, _FAST["mask"].size - 1, out=idx)
    scores = _FAST["score"][idx].max(axis=1)
    n = boards.shape[0]
    return scores[:n], scores[n:]


def _deck_without(cards):
    """The 36-card deck minus the cards already seen, as a numpy array."""
    used = _np.zeros(36, dtype=bool)
    used[list(cards)] = True
    return _np.flatnonzero(~used)


def _mc_equity(hole, board, trials, offset):
    """Win rate (ties count half) versus one uniformly random opponent hand."""
    deck = _deck_without(list(hole) + list(board))
    pool = _FAST["pools"].get(deck.size)
    if pool is None:
        return 0.5
    trials = max(8, min(int(trials), POOL_SIZE))
    n_extra = 5 - len(board)
    draws = 5 + n_extra
    rows = pool[(offset + _np.arange(trials)) % POOL_SIZE][:, :draws]
    cards = deck[rows]                       # (trials, draws) distinct cards
    opps = cards[:, :5]
    known = _np.tile(_np.asarray(board, dtype=_np.int64), (trials, 1))
    boards = _np.concatenate((known, cards[:, 5:]), axis=1) if n_extra else known
    mine, theirs = _paired_scores(hole, opps, boards)
    return float((mine > theirs).sum() + 0.5 * (mine == theirs).sum()) / trials


def _mc_equity_py(hole, board, trials, rng):
    """Same idea without numpy: far slower, so far fewer samples."""
    seen = set(hole) | set(board)
    deck = [c for c in range(36) if c not in seen]
    need = 5 - len(board)
    score = 0.0
    for _ in range(trials):
        pick = rng.sample(deck, 5 + need)
        full = list(board) + pick[5:]
        mine = _omaha_score(list(hole), full)
        theirs = _omaha_score(pick[:5], full)
        score += 1.0 if mine > theirs else (0.5 if mine == theirs else 0.0)
    return score / trials


# Preflop: a 14-feature hand model fitted offline against 1.8M simulated
# showdowns (6,000 random hands x 300 deals).  It predicts equity against a
# random hand with R^2 = 0.71, and its extremes are calibrated (worst 5%:
# predicted 0.373 vs actual 0.377; best 5%: 0.619 vs 0.617).
# features: pair, two_pair, trips, ace, high, top1, top2, suited, double_suited,
#           gap1, gap2, gap34, high_pair, pair_rank
_PF_COEF = (0.298329, 0.007212, 0.044005, -0.132992, 0.001285, 0.111499,
            -0.047507, 0.079186, -0.075714, 0.030141, 0.090271, 0.099290,
            0.108768, -0.044693, 0.129344)


def _preflop_equity(hole):
    """Cheap, noise-free preflop strength estimate (no simulation needed)."""
    ranks = sorted((c >> 2 for c in hole), reverse=True)
    counts = {}
    for r in ranks:
        counts[r] = counts.get(r, 0) + 1
    suits = {}
    for c in hole:
        s = c & 3
        suits[s] = suits.get(s, 0) + 1
    sv = sorted(suits.values(), reverse=True)
    pairs = [r for r, v in counts.items() if v >= 2]
    g1 = g2 = g34 = 0
    for i in range(5):
        for j in range(i + 1, 5):
            gap = abs(ranks[i] - ranks[j])
            gap = min(gap, 9 - gap)          # the ace also plays low (A-6-7-8-9)
            if gap == 1:
                g1 += 1
            elif gap == 2:
                g2 += 1
            elif gap <= 4:
                g34 += 1
    x = (
        1.0 if pairs else 0.0,
        1.0 if len(pairs) >= 2 else 0.0,
        1.0 if any(v >= 3 for v in counts.values()) else 0.0,
        1.0 if 8 in counts else 0.0,
        sum(1 for r in ranks if r >= 5) / 5.0,
        ranks[0] / 8.0,
        (ranks[0] + ranks[1]) / 16.0,
        sum(c * (c - 1) // 2 for c in sv) / 10.0,
        1.0 if (len(sv) > 1 and sv[0] == 2 and sv[1] == 2) or sv[0] >= 4 else 0.0,
        g1 / 10.0, g2 / 10.0, g34 / 10.0,
        float(sum(1 for r in pairs if r >= 5)),
        (max(pairs) / 8.0) if pairs else 0.0,
    )
    equity = _PF_COEF[0]
    for coef, value in zip(_PF_COEF[1:], x):
        equity += coef * value
    return min(max(equity, 0.20), 0.80)


# How much equity we need, by street: (to raise facing a bet, to bet/raise when
# checked to).  Short stacks (40 bb) mean thin edges are not worth the variance.
_RAISE_EQ = {Street.PREFLOP: 0.60, Street.FLOP: 0.66,
             Street.TURN: 0.70, Street.RIVER: 0.74}
_BET_EQ = {Street.PREFLOP: 0.56, Street.FLOP: 0.60,
           Street.TURN: 0.64, Street.RIVER: 0.68}
_POT_RAISE_EQ = {Street.PREFLOP: 0.62, Street.FLOP: 0.76,
                 Street.TURN: 0.80, Street.RIVER: 0.78}
# Extra equity required on top of the raw pot odds before we call.  It is higher
# on later streets because a betting opponent holds better than a random hand.
_CALL_MARGIN = {Street.PREFLOP: 0.00, Street.FLOP: 0.02,
                Street.TURN: 0.04, Street.RIVER: 0.06}
_BLUFF = {Street.PREFLOP: 0.05, Street.FLOP: 0.10,
          Street.TURN: 0.08, Street.RIVER: 0.05}


def _trial_count(obs, memory):
    """Spend a fair share of what is left of the time bank on this decision."""
    budget = memory["budget"]
    left = budget["bank"] * BANK_SHARE - budget["spent"]
    if left <= 0.0:
        return 8
    hands_left = max(budget["hands"] - obs.hand_index, 1)
    # ~2.2 simulated decisions per hand on average; decisions we skip (preflop)
    # simply leave more bank for the rest of the half.
    allowance = left / (hands_left * 2.2)
    allowance = min(max(allowance, 0.00018), 0.0032)
    # The cost of a trial is measured as we go, so a slower tournament machine
    # automatically gets fewer simulations instead of an empty time bank.
    trials = int(allowance / memory["stats"]["trial_cost"])
    if obs.to_call >= 8:            # a big decision: a bet of 4 bb or more
        trials = int(trials * 1.35)
    return max(MIN_TRIALS, min(MAX_TRIALS, trials))


def _track_opponent(obs, stats):
    """Count how often the opponent has raised during this half."""
    if obs.hand_index != stats["hand_seen"]:
        stats["hand_seen"] = obs.hand_index
        stats["hand_opp_raises"] = 0
    seen = 0
    for record in obs.action_history:
        if not record.is_me and record.action in (Action.RAISE_HALF, Action.RAISE_POT):
            seen += 1
    if seen > stats["hand_opp_raises"]:
        stats["opp_raises"] += seen - stats["hand_opp_raises"]
        stats["hand_opp_raises"] = seen


def _raise(obs, equity, street, cheap=False):
    """Pick a raise size: pot with a monster, half pot otherwise."""
    legal = obs.legal_actions
    if not cheap and equity >= _POT_RAISE_EQ[street] and Action.RAISE_POT in legal:
        return Action.RAISE_POT
    if Action.RAISE_HALF in legal:
        return Action.RAISE_HALF
    if Action.RAISE_POT in legal:
        return Action.RAISE_POT
    return Action.CHECK_CALL


def _pick(obs, equity, memory):
    """Turn an equity estimate into a legal action."""
    legal = obs.legal_actions
    street = obs.street
    stats = memory["stats"]
    hands = stats["hands"]

    # Opponent tendencies, shrunk towards neutral priors so early hands do not
    # overreact to a single observation.
    aggression = (stats["opp_raises"] + 0.35) / (hands + 1.0)          # raises/hand
    fold_rate = (stats["opp_folded"] + 3.0) / (stats["our_bets"] + 8.0)

    bet_eq = _BET_EQ[street]
    raise_eq = _RAISE_EQ[street]
    margin = _CALL_MARGIN[street] - 0.035 * min(aggression, 1.2)
    if fold_rate < 0.30:          # a station: bet thinner, never bluff
        bet_eq -= 0.03
        raise_eq -= 0.02
    bluff = _BLUFF[street] * max(0.0, min((fold_rate - 0.22) / 0.35, 1.2))

    can_raise = obs.raises_this_street < 3 and (
        Action.RAISE_HALF in legal or Action.RAISE_POT in legal)

    if obs.to_call <= 0:
        if can_raise:
            if equity >= bet_eq:
                return _raise(obs, equity, street)
            if equity >= 0.28 and memory["rng"].random() < bluff:
                return _raise(obs, 0.0, street, cheap=True)
        return Action.CHECK_CALL

    pot_odds = obs.to_call / (obs.pot + obs.to_call)
    if can_raise and equity >= raise_eq:
        return _raise(obs, equity, street)
    if equity >= pot_odds + margin:
        return Action.CHECK_CALL
    return Action.FOLD if Action.FOLD in legal else Action.CHECK_CALL


def _choose(obs, memory):
    """Estimate equity, answer, and charge the time spent to the budget."""
    started = _time.perf_counter()
    stats = memory["stats"]
    _track_opponent(obs, stats)

    if obs.street == Street.PREFLOP:
        equity = _preflop_equity(obs.hole_cards)
    else:
        trials = _trial_count(obs, memory)
        drew_at = _time.perf_counter()
        if _FAST["ready"]:
            equity = _mc_equity(obs.hole_cards, obs.board, trials, stats["pool_offset"])
        else:
            trials = max(8, trials // 3)
            equity = _mc_equity_py(obs.hole_cards, obs.board, trials, memory["rng"])
        stats["pool_offset"] = (stats["pool_offset"] + trials + 11) % POOL_SIZE
        # Track the real cost of one trial (exponentially weighted) so the
        # budget above adapts to the speed of whatever machine we run on.
        cost = (_time.perf_counter() - drew_at) / trials
        stats["trial_cost"] = min(max(0.85 * stats["trial_cost"] + 0.15 * cost,
                                     0.000003), 0.0004)

    action = _pick(obs, equity, memory)
    if action in (Action.RAISE_HALF, Action.RAISE_POT):
        stats["we_bet"] = True
    memory["budget"]["spent"] += _time.perf_counter() - started
    return action


def _fallback(obs):
    """The belt-and-braces answer if anything at all goes wrong."""
    if Action.CHECK_CALL in obs.legal_actions:
        return Action.CHECK_CALL
    return Action.FOLD


def _reset(memory, config):
    """Start of a half: rebuild every piece of state, including the budget."""
    bank = getattr(config, "time_bank_seconds", None) or 8.0
    hands = getattr(config, "num_hands", None) or 1000
    memory["budget"] = {"bank": float(bank), "hands": int(hands), "spent": 0.0}
    memory["stats"] = {
        "hands": 0,               # finished hands this half
        "opp_raises": 0,          # opponent raises seen this half
        "hand_seen": -1,
        "hand_opp_raises": 0,
        "our_bets": 0,            # hands where we put in a raise
        "opp_folded": 0,          # ... of which the opponent folded
        "we_bet": False,
        "pool_offset": 0,
        "trial_cost": SEC_PER_TRIAL,   # seconds per simulated showdown, re-measured
    }
    memory["rng"] = _random.Random(12345)


def _learn(result, memory):
    """One finished hand: keep the opponent model up to date."""
    stats = memory["stats"]
    stats["hands"] += 1
    if stats["we_bet"]:
        stats["our_bets"] += 1
        if not result.went_to_showdown and result.winner == "me":
            stats["opp_folded"] += 1
    stats["we_bet"] = False


def new_half(config: AgentConfig, memory: dict) -> None:
    """Called at the start of each half-match (1000 hands).

    `memory` is a fresh, empty dict every half. Store anything you want to
    remember between hands in it, e.g. memory["opp_raises"] = 0.
    `config` holds the match settings (starting_stack, big_blind, ...).

    In the tournament, each half runs in a brand-new copy of your server,
    so nothing you learn in one half carries over to the other half.
    """
    _reset(memory, config)


def choose_action(obs: Observation, memory: dict) -> Action:
    """Called every time it is your turn. Return one of obs.legal_actions.

    What you can see in obs:
        hand_index          which hand of the half this is (0, 1, 2, ...)
        street              Street.PREFLOP, Street.FLOP, Street.TURN or Street.RIVER
        hole_cards          your 5 cards as ints, sorted ascending (0-35)
        board               board (community) cards seen so far (0, 3, 4 or 5 ints)
        pot                 total chips both players put in this hand
        to_call             chips you must add to match the opponent
        my_bet, opp_bet     chips you,opponent put in on THIS street
        my_stack, opp_stack chips each player still has behind
        is_button           True if you are the button (small blind)
        legal_actions       the Actions you may return right now
        raise_to            {RAISE_HALF: total, RAISE_POT: total} for legal raises
        min_raise_to        smallest normal raise total (all-ins may be lower)
        raises_this_street  raises so far this street (max 4)
        action_history      this hand's actions so far; each has
                            street, is_me, action, amount, street_total
        big_blind           size of the big blind in chips

    Actions: Action.FOLD, Action.CHECK_CALL, Action.RAISE_HALF, Action.RAISE_POT
    Cards are ints: card_rank(c) is 0 (six) .. 8 (ace), card_suit(c) is 0..3,
    and cards_to_str(obs.hole_cards) prints them like "6h 7c 9s Kd Ah".
    Once all 5 board cards are out, omaha_score(obs.hole_cards, obs.board)
    scores your hand (higher is better).
    """

    # Any mistake inside _choose (or in a library) costs an error, and five
    # errors forfeit the whole half - so never let an exception escape.
    try:
        return _choose(obs, memory)
    except Exception:
        return _fallback(obs)


def on_hand_end(result: HandResult, memory: dict) -> None:
    """Called once for every finished hand. Optional: learn from what happened.

    To save time, the judge sends finished hands along with your next
    decision request, so this always runs BEFORE choose_action() sees the
    next hand. (The last hands of a half arrive at the end of the half.)

    result has: hand_index, my_net (chips won or lost), pot,
    went_to_showdown, board (cards revealed), opp_hole (opponent's cards,
    only after a showdown, else None), winner ('me', 'opp' or 'split').
    """
    _learn(result, memory)

# ===== END OF YOUR CODE =====


# ===== SERVER PLUMBING (you normally don't need to edit below) =====
# The judge calls these routes:
#   GET  /           -> who you are (also used as the health check)
#   POST /new-half   -> start a fresh half: memory is wiped
#   POST /act        -> the judge sends any finished hands plus an Observation;
#                       on_hand_end() runs for each hand, then you answer an action
#   POST /hand-end   -> the judge sends the last finished hands of a half
#
# Why "async def"? The routes run on one thread and call your functions
# directly, so your functions never run twice at once. Your own functions
# stay normal "def" (don't make them async).
#
# Keep your per-half state in `memory`. In the tournament every half runs in
# a fresh server process, so everything starts over. When you test locally,
# though, one process can play both halves and only `memory` is wiped, NOT
# other global variables. Using `memory` makes your bot behave the same way
# in both places.

app = FastAPI()
MEMORY: dict = {}                                   # your memory for this half
STATUS = {"started": False}                         # has /new-half been called?
TOKEN = os.environ.get("AGENT_TOKEN") or None       # set by the tournament


async def _read_request(request: Request, path: str) -> tuple[dict, JSONResponse | None]:
    """Check the token (and that the half has started) BEFORE reading the body.

    Checking first means a request without the right token costs your bot
    almost nothing, so nobody else can slow it down by sending junk.
    Returns (body, None) when the request is fine, or ({}, error response).
    """
    if TOKEN and request.headers.get("x-agent-token") != TOKEN:
        return {}, JSONResponse({"error": "bad or missing X-Agent-Token"}, status_code=401)
    if not STATUS["started"] and path != "/new-half":
        return {}, JSONResponse({"error": "call /new-half first"}, status_code=409)
    try:
        body = await request.json()
    except ValueError:
        body = None
    if not isinstance(body, dict):
        return {}, JSONResponse({"error": "the request body must be a JSON object"}, status_code=400)
    return body, None


def _error_info(exc: Exception) -> dict:
    """Describe an exception from your code (call this inside `except`)."""
    return {"error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}


def _crash(exc: Exception, hand_end_errors: list | None = None) -> JSONResponse:
    """Report an exception from your code to the judge (with traceback)."""
    return JSONResponse({**_error_info(exc), "hand_end_errors": hand_end_errors or []},
                        status_code=500)


def _deliver_results(raw_results) -> list:
    """Call on_hand_end() for each finished hand the judge sent.
    Returns a list of error reports (one per hand that crashed)."""
    errors = []
    for raw in raw_results or []:
        try:
            on_hand_end(HandResult.from_dict(raw), MEMORY)
        except Exception as exc:
            errors.append(_error_info(exc))
    return errors


@app.get("/")
async def info():
    """Health check: tells the judge who this bot is."""
    return {"participant": PARTICIPANT, "agent_name": AGENT_NAME}


@app.post("/new-half")
async def route_new_half(request: Request):
    """Wipe memory and call your new_half(). The body is {"config": {...}}."""
    body, refused = await _read_request(request, "/new-half")
    if refused:
        return refused
    MEMORY.clear()
    STATUS["started"] = True
    try:
        new_half(AgentConfig.from_dict(body.get("config", {})), MEMORY)
    except Exception as exc:
        return _crash(exc)
    return {"ok": True}


@app.post("/act")
async def route_act(request: Request):
    """Deliver finished hands, then turn the JSON into an Observation, call
    choose_action(), and send the answer back."""
    body, refused = await _read_request(request, "/act")
    if refused:
        return refused
    hand_end_errors = _deliver_results(body.get("hand_results"))
    try:
        obs = Observation.from_dict(body["observation"])
        action = choose_action(obs, MEMORY)
    except Exception as exc:
        return _crash(exc, hand_end_errors)
    # Send the action's name; anything strange is passed on for the judge to reject.
    if isinstance(action, Action):
        answer = action.name
    elif action is None or isinstance(action, (str, int, float, bool)):
        answer = action
    else:
        answer = repr(action)
    return {"action": answer, "hand_end_errors": hand_end_errors}


@app.post("/hand-end")
async def route_hand_end(request: Request):
    """Call on_hand_end() for the last finished hands of a half."""
    body, refused = await _read_request(request, "/hand-end")
    if refused:
        return refused
    return {"ok": True, "hand_end_errors": _deliver_results(body.get("results"))}


if __name__ == "__main__":
    # The tournament sets PORT as an environment variable, so each bot is compliant to our evaluation platform
    # Locally it defaults to 5008 for your own testing convenience
    port = int(os.environ.get("PORT", "5008"))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info", access_log=False)
