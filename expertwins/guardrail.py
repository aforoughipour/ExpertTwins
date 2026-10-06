"""The heterogeneity band, and what to do when the table falls out of it.

THE REQUIREMENT, IN THE OPERATOR'S WORDS: a multi-agent system that converges to
one statement after a round of discussion has failed, and the failure must be
prevented rather than merely noticed. There must always be some level of
difference between the experts; the level depends on who is at the table and
what is being discussed; and when the difference falls below it the agents must
be pushed back toward their own frames.

THE THING BEING DEFENDED AGAINST IS MEASURED, AND IT IS FAST.

    "when agents read each other's complete outputs, their proposals converge
     within one round, erasing the diversity that motivates using multiple
     models"
        -- The Interaction Tax, Ann, Liu & Tan, ICML 2026, arXiv:2608.23541
           (VERIFIED; see docs/06-measurement.md)

Independent generation avoids it. Conformity is strongest exactly where it costs
most: LLMs conform more when they are individually uncertain (arXiv:2410.12428,
VERIFIED), which is to say on contested scientific questions. And preserving
within-session agent diversity has been shown to be a NECESSARY condition for
diverse outputs, with two mechanisms that work -- anchoring each agent to a
persistent distinct mode, and restricting each agent's context to its most
semantically DISTANT peers (arXiv:2609.00683, VERIFIED).

THE HARD PART IS NOT THE PUSH. IT IS KNOWING WHEN NOT TO.

    COLLAPSE     the seats converge because they read each other, not because
                 the science converged.
    MANUFACTURE  the band forces dissent on a question where real experts agree.
                 Worse than collapse, because collapse is visible in the
                 transcript and manufactured dissent is not. And it is not
                 hypothetical: debate-framing instructions measurably INCREASE
                 persona inconstancy (arXiv:2405.03862, VERIFIED), so pushing
                 harder degrades the identities being protected.

Hence a TWO-SIDED BAND rather than a floor -- the form is transferred from
entropy-band regularization in test-time RL (arXiv:2511.17938, VERIFIED) -- and
hence a mechanical distinction this module can actually make:

    same conclusion + DIFFERENT evidence  ->  independent corroboration, the
                                              best thing a panel can produce.
                                              Do nothing.
    same conclusion + SAME evidence       ->  collapse. Intervene.
    different conclusions + same evidence ->  arguing over one text. Healthy.
                                              Do nothing.

THE REFERENCE IS TURN 1, AND THAT IS THE DESIGN'S ONE REAL IDEA.

Turn 1 is generated in isolation: no seat has seen another seat's anything. It
is therefore the only available measurement of what these particular seats, on
this particular question, produce when they cannot influence each other -- and
it already encodes both things the threshold must depend on, because a settled
question yields a low turn-1 separation even from maximally distinct scientists.

    band_n = [rho_low * S_1,  rho_high * S_1]

`rho_low` is raised for contested questions, because that is where conformity
pressure is highest. No published work sets a diversity floor this way. The
components are cited; the construction is ours; it is labelled as such.

INTERVENTIONS ACT ON EVIDENCE AND ROUTING BEFORE THEY ACT ON INSTRUCTIONS.

Telling an agent to disagree produces text that disagrees. The ladder is ordered
so every rung that changes *what a seat reads or sees* is tried before the one
rung that changes *what a seat is told*, and that rung stamps the turn as
prompt-perturbed so a reader can discount it. If the ladder is exhausted and the
table is still collapsed, the run says so. It never fakes a recovery.

FIDELITY IS A CONSTRAINT ON THE WHOLE LOOP. A diversity gain bought with a
fidelity loss is refused: without that constraint the band rewards seats for
saying things the scientist would never say, which is an identity failure.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum

from .diversity import (
    CALIBRATED_BACKEND, Preflight, SeatTurn, Snapshot, consensus_docs,
)

#: Absolute floors. Below these a table is indistinguishable from one agent no
#: matter what turn 1 looked like -- if turn 1 was itself degenerate, a purely
#: relative band would happily certify the degeneracy.
#:
#: THE EVIDENCE FLOOR IS BACKEND-INDEPENDENT. The evidence axis is a Jaccard
#: statistic over cited document ids; no embedding touches it, so one constant
#: is correct for every backend.
ABS_EVIDENCE_FLOOR = 0.15

#: THE POSITIONS FLOOR IS A PROPERTY OF THE EMBEDDING BACKEND, AND TREATING IT
#: AS A PROPERTY OF THE CONSTRUCT IS A BUG THIS CODE ONCE HAD.
#:
#: `positions` is `(VS2 - 1) / (m - 1)` over embedded claim text, so its scale
#: is the scale of the kernel that produced it. A single fixed floor across
#: backends would certify the same text under one kernel and stamp it
#: degenerate under another. The compression is a property of long concatenated
#: domain-homogeneous text, where sentence-transformer embeddings occupy a
#: narrow cone, not of the model in the abstract. A synthetic test panel of
#: short distinct sentences will therefore never reveal it.
#:
#: PROVENANCE OF THE MiniLM NUMBER, STATED PLAINLY: it is an extrapolation, not
#: a calibration. It has not been validated against a labelled collapse, because
#: no such set exists here.
#:
#: tfidf-cosine REMAINS THE RECOMMENDED BACKEND for this measurement, and not
#: for compatibility. It gives the semantic axis more discriminating headroom
#: for this use case.
ABS_POSITIONS_FLOORS: dict[str, float] = {
    "tfidf-cosine": 0.20,
    "sentence-transformers:all-MiniLM-L6-v2": 0.023,
}

#: The recommended backend's value, kept under its original name because it is
#: referenced as a constant elsewhere and in the manual.
ABS_POSITIONS_FLOOR = ABS_POSITIONS_FLOORS["tfidf-cosine"]


def abs_positions_floor(backend: str) -> tuple[float, bool]:
    """Absolute positions floor for one backend: (floor, calibrated?).

    An unrecognised backend returns (0.0, False) -- NO absolute floor, and the
    caller is told the check could not be made rather than being handed a number
    that silently came off a different instrument. A threshold breach measured
    on an uncalibrated instrument is worse than no breach at all, because it
    reads exactly like a finding.
    """
    hit = ABS_POSITIONS_FLOORS.get(backend or "")
    return (hit, True) if hit is not None else (0.0, False)


#: Below this, "new" claims are restatements of what is already on the table.
NOVELTY_FLOOR = 0.25
#: A seat on its own territory that anchors nothing in its own papers has
#: stopped being that person; a diversity gain bought at this price is refused.
FIDELITY_FLOOR = 0.25
#: Below this share of the invited roster, no heterogeneity verdict is made.
#: Dropping the seats that agree with each other raises every statistic here.
MIN_PARTICIPATION = 0.6

#: Ceiling coefficient. AN ENGINEERING DEFAULT, NOT A PUBLISHED VALUE.
RHO_HIGH = 1.20


class Verdict(str, Enum):
    HEALTHY = "healthy"
    CORROBORATION = "independent_corroboration"
    CONTESTED_EVIDENCE = "contested_evidence"
    COLLAPSING = "collapsing"
    COLLAPSED = "collapsed"
    #: The turn the band was fixed from. It cannot breach a band it defines;
    #: what it CAN do is start below the absolute floors, which is a different
    #: and more serious finding -- a panel that was degenerate before anyone
    #: spoke to anyone.
    REFERENCE = "reference_turn"
    #: Above the band: more scatter than the isolated first turn produced, which
    #: is not a triumph -- it is what manufactured dissent looks like.
    SCATTERED = "scattered"
    #: A seat stopped being the person it is supposed to be. Nothing about this
    #: turn's heterogeneity is interpretable until that is fixed, so this
    #: SHORT-CIRCUITS the verdict rather than annotating it.
    FIDELITY_LOST = "fidelity_lost"
    #: Not enough claims to say anything. Reported, never silently treated as
    #: healthy: an empty table is not a diverse one.
    UNDETERMINED = "undetermined"


class Move(str, Enum):
    """The ladder, in the order it is climbed."""

    NONE = "none"
    DISTANT_PEERS = "route_to_most_distant_peers"
    DIFFERENTIAL_RETRIEVAL = "withhold_the_consensus_evidence"
    REANCHOR = "reanchor_to_own_corpus"
    ANONYMIZE = "anonymize_the_table"
    ISOLATE = "isolate_from_cross_talk"
    ASSIGN_DISSENT = "assign_dissent_to_the_evidenced_minority"
    TERMINATE = "stop_deliberating_and_report"
    DECLARE = "declare_collapse"


#: What each rung does and what it rests on, for the transcript. A reader must
#: be able to see that the panel was perturbed, how, and on whose authority --
#: otherwise the perturbation becomes an invisible confound.
MOVE_DESCRIPTION = {
    Move.NONE: "No intervention.",
    Move.DISTANT_PEERS: (
        "Each seat was shown only the claims of the seats furthest from it in "
        "embedding space, rather than the whole table. This counters majority "
        "pull by removing the majority from the seat's context. Acts on "
        "routing; does not touch the prompt. [arXiv:2609.00683, "
        "Embedding-based Peer Selection -- VERIFIED]"),
    Move.DIFFERENTIAL_RETRIEVAL: (
        "The documents the table had converged on were withheld from the next "
        "round of retrieval, pushing every seat onto evidence the others were "
        "not using. Acts on evidence; does not touch the prompt. [our "
        "construction; rests on corpus exclusivity being the live axis of "
        "heterogeneity]"),
    Move.REANCHOR: (
        "Each seat's evidence was restricted to papers it authored, forcing the "
        "scientist back onto their own results, where their distinctiveness "
        "actually lives. Acts on evidence; does not touch the prompt. "
        "[arXiv:2609.00683 Cognitive Lens Assignment; arXiv:2402.10962 -- "
        "instruction instability grows with dialogue length -- both VERIFIED]"),
    Move.ANONYMIZE: (
        "Seat names were stripped from the claims shown to the table, so a "
        "position could not be deferred to because of who held it. Acts on "
        "presentation; does not touch the prompt. [arXiv:2510.07517 -- "
        "sycophancy dominates self-bias, and anonymisation measurably reduces "
        "it -- VERIFIED]"),
    Move.ISOLATE: (
        "Cross-talk was removed entirely for one turn: no seat was shown any "
        "other seat's claims. This is the regime in which the convergence was "
        "NOT observed. Acts on the channel; does not touch the prompt. "
        "[arXiv:2608.23541 -- VERIFIED]"),
    Move.ASSIGN_DISSENT: (
        "The seat whose OWN evidence best supports the minority position was "
        "required to argue it. THIS RUNG PERTURBS THE PROMPT, and it is placed "
        "this late deliberately: debate-framing instructions increase persona "
        "inconstancy, so this rung trades fidelity for diversity. Disagreement "
        "produced under it is weaker evidence of genuine divergence and must be "
        "read that way. [arXiv:2410.12428 -- devil's advocate mitigates "
        "conformity; arXiv:2405.03862 -- the caution; both VERIFIED]"),
    Move.TERMINATE: (
        "Deliberation stopped and the positions were reported as they stood. "
        "Further rounds are not neutral: more rounds before aggregation reduce "
        "performance, while more agents help. [arXiv:2502.19130 -- VERIFIED]"),
    Move.DECLARE: (
        "The ladder was exhausted and the table did not recover. The run is "
        "marked collapsed. Nothing was faked to hide it."),
}

#: Routing and evidence rungs first; the prompt rung late; the admission last.
LADDER = [Move.DISTANT_PEERS, Move.DIFFERENTIAL_RETRIEVAL, Move.REANCHOR,
          Move.ANONYMIZE, Move.ISOLATE, Move.ASSIGN_DISSENT, Move.TERMINATE,
          Move.DECLARE]


@dataclass
class Band:
    """The thresholds for one run, fixed at turn 1 and never renegotiated."""

    rho_low: float
    rho_high: float
    evidence: float
    positions: float
    ceiling_positions: float
    #: The turn-1 separations themselves, kept so the retention statistic is
    #: `S_now / S_1` rather than `S_now / floor`. Those differ whenever an
    #: ABSOLUTE floor binds; reporting the second while calling it the first
    #: would make the retention statistic unreadable.
    ref_evidence: float = 0.0
    ref_positions: float = 0.0
    #: Which instrument's absolute floors were applied, and whether that
    #: instrument had any. A band whose `abs_calibrated` is False is a purely
    #: RELATIVE band: it can still detect a panel that collapsed relative to its
    #: own first turn, but it cannot detect a panel that was degenerate from the
    #: start, and it must not be read as having checked for one.
    abs_backend: str = "tfidf-cosine"
    abs_calibrated: bool = True
    #: What this band describes. A band is a statement about ONE panel on ONE
    #: question measured with ONE embedding backend; comparing across any of
    #: those is meaningless, so the identity travels with it.
    identity: dict | None = None
    reference_turn: int = 1
    source: str = ""

    def render(self) -> str:
        abs_note = (f"calibrated on {self.abs_backend}" if self.abs_calibrated
                    else f"NO absolute floor: {self.abs_backend!r} is "
                         f"uncalibrated, so this band is RELATIVE ONLY and has "
                         f"not checked for a degenerate reference turn")
        return (f"band (rho_low={self.rho_low:.2f}, rho_high={self.rho_high:.2f} "
                f"of turn {self.reference_turn})\n"
                f"  floor    evidence {self.evidence:.3f}   positions "
                f"{self.positions:.3f}\n"
                f"  ceiling  positions {self.ceiling_positions:.3f}\n"
                f"  abs      {abs_note}\n"
                f"  identity {self.identity}\n"
                f"  {self.source}")

    def to_json(self) -> dict:
        return {"rho_low": self.rho_low, "rho_high": self.rho_high,
                "evidence": self.evidence, "positions": self.positions,
                "ceiling_positions": self.ceiling_positions,
                "ref_evidence": self.ref_evidence,
                "ref_positions": self.ref_positions,
                "abs_backend": self.abs_backend,
                "abs_calibrated": self.abs_calibrated,
                "identity": self.identity,
                "reference_turn": self.reference_turn, "source": self.source}


def set_band(pre: Preflight, first: Snapshot) -> Band:
    """Fix the band from the preflight prior and the isolated first turn.

    Fixed ONCE. A band recomputed each turn from the current turn would ratchet
    downward with the conversation and certify any collapse that happened
    slowly -- which is the only way collapse ever actually happens.

    THE LEXICAL AXIS HAS NO FLOOR HERE, DELIBERATELY. It is reported as a style
    channel and nothing else: form-based diversity metrics assign high diversity
    to sets of random sentences (arXiv:2506.00514). Setting a lexical floor and
    appending its breach to the verdict would contradict the module's own
    documentation.
    """
    rho = pre.retention
    abs_pos, calibrated = abs_positions_floor(first.backend)
    abs_note = ("" if calibrated else
                f" NO ABSOLUTE POSITIONS FLOOR WAS APPLIED: {first.backend!r} "
                f"has no calibration in ABS_POSITIONS_FLOORS, and the floors on "
                f"record came off instruments whose similarity scale is not "
                f"this one's. The positions side of this band is RELATIVE ONLY "
                f"-- it can see a panel fall away from its own first turn, and "
                f"it CANNOT see a panel that was degenerate to begin with.")
    return Band(
        rho_low=rho,
        rho_high=RHO_HIGH,
        evidence=max(ABS_EVIDENCE_FLOOR, rho * first.s("evidence")),
        positions=max(abs_pos, rho * first.effective_positions),
        ceiling_positions=min(1.0, max(0.35, RHO_HIGH * first.effective_positions)),
        ref_evidence=first.s("evidence"),
        ref_positions=first.effective_positions,
        abs_backend=first.backend,
        abs_calibrated=calibrated,
        reference_turn=first.turn,
        source=(
            f"turn {first.turn} was generated in isolation, so its separation "
            f"(evidence {first.s('evidence'):.3f}, positions "
            f"{first.effective_positions:.3f}, VS2 {first.vendi2:.2f} of "
            f"{first.vendi_max}) is this panel's free diversity on this "
            f"question -- ONE stochastic draw of it, not a population value. "
            f"rho_low={rho:.2f}, from packet separation "
            f"{pre.packet_separation:.2f} and contestedness "
            f"{pre.contestedness:.2f}; rho_high={RHO_HIGH:.2f} is a fixed "
            f"convention guarding against manufactured scatter."
            f"{abs_note}"),
    )


# --------------------------------------------------------------------------
# the sequential monitor
# --------------------------------------------------------------------------

@dataclass
class Sequential:
    """A heuristic sequential accumulator over the band statistic.

    WHY NOT SIMPLY THRESHOLD EACH TURN. With ten seats and a handful of claims
    each, one turn's separation is noisy; a per-turn threshold will both fire
    spuriously and miss slow drift, and a guardrail that fires spuriously is
    itself a confound. Accumulating across turns fixes both.

    This is not a calibrated Wald SPRT. The shape is borrowed from the compute governor of
    arXiv:2605.19193 -- accumulate a log-likelihood ratio under a Beta family,
    stop at a Wald boundary -- but the assumptions a Wald SPRT needs are all
    violated here:

      * the observations are serially dependent by construction: the same seats
        discussing the same material, each turn conditioned on the last
      * the Beta parameters below are engineering choices, not fitted
        likelihoods, so the boundaries are not nominal error rates
      * the policy CHANGES when the monitor fires, so the post-firing
        observations do not come from the pre-firing model
      * the statistic is measured against a single noisy baseline

    So the boundaries are TUNING CONSTANTS rather than error rates. In practice
    they encode: **one unambiguous turn fires immediately** (a fully collapsed
    axis sends the increment to the boundary in a single step, which is right --
    total collapse is not a noisy observation), while a mildly degraded turn
    needs several consecutive siblings. Anyone wanting real sequential
    guarantees needs a calibrated change-point model or an anytime-valid
    procedure whose conditional model tolerates adaptive intervention. Recorded
    rather than papered over.

    The accumulator is reset on firing because the regime it was testing has
    just been deliberately changed -- which is the honest thing to do with a
    statistic whose generating process no longer holds, and is also why no
    error-rate claim survives the reset.
    """

    llr: float = 0.0
    upper: float = math.log(0.95 / 0.05)
    lower: float = math.log(0.05 / 0.95)
    history: list[float] = field(default_factory=list)

    @staticmethod
    def _beta_logpdf(x: float, a: float, b: float) -> float:
        x = min(max(x, 1e-6), 1 - 1e-6)
        return ((a - 1) * math.log(x) + (b - 1) * math.log(1 - x)
                + math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b))

    def update(self, retained: float | None) -> str:
        """Feed one turn's band statistic; return 'collapsing' | 'healthy' | ''.

        `retained` is the share of turn-1 separation still present, in [0, 1],
        or **None when no axis could be measured**. None is a MISSING
        OBSERVATION and the accumulator does not move: feeding 1.0 in that case
        would score an unmeasurable turn as perfect health.
        """
        if retained is None:
            self.history.append(round(self.llr, 4))
            return ""
        x = min(max(retained, 0.0), 1.0)
        self.llr += (self._beta_logpdf(x, 2.0, 6.0)      # H1: collapsing
                     - self._beta_logpdf(x, 6.0, 2.0))   # H0: healthy
        self.history.append(round(self.llr, 4))
        if self.llr >= self.upper:
            self.llr = 0.0
            return "collapsing"
        if self.llr <= self.lower:
            self.llr = 0.0
            return "healthy"
        return ""

    def to_json(self) -> dict:
        return {"llr": self.llr, "upper": self.upper, "lower": self.lower,
                "history": self.history}

    @classmethod
    def from_json(cls, d: dict | None) -> "Sequential":
        if not d:
            return cls()
        s = cls(llr=float(d.get("llr", 0.0)))
        s.history = list(d.get("history", []))
        return s


# --------------------------------------------------------------------------
# the assessment
# --------------------------------------------------------------------------

@dataclass
class Assessment:
    turn: int
    verdict: Verdict
    breached: list[str] = field(default_factory=list)
    move: Move = Move.NONE
    reasons: list[str] = field(default_factory=list)
    exclude_docs: set[str] = field(default_factory=set)
    #: Which seat was asked to argue the minority position, for ASSIGN_DISSENT.
    dissenter: str = ""
    #: Rungs skipped because they could not bite. The caller advances the ladder
    #: by 1 + this, so a rung that cannot apply is not retried forever.
    skipped_rungs: int = 0
    llr: float = 0.0
    monitor: dict = field(default_factory=dict)

    @property
    def intervene(self) -> bool:
        return self.move is not Move.NONE

    def render(self) -> str:
        lines = [f"turn {self.turn}: {self.verdict.value.upper()}"]
        for r in self.reasons:
            lines.append(f"  - {r}")
        if self.breached:
            lines.append(f"  breached: {', '.join(self.breached)}")
        lines.append(f"  sequential LLR {self.llr:+.2f} "
                     f"(fires at {Sequential().upper:+.2f})")
        if self.intervene:
            lines.append(f"  MOVE: {self.move.value}")
            lines.append(f"    {MOVE_DESCRIPTION[self.move]}")
            if self.exclude_docs:
                lines.append(f"    withholding {len(self.exclude_docs)} document(s)")
        return "\n".join(lines)

    def to_json(self) -> dict:
        return {"turn": self.turn, "verdict": self.verdict.value,
                "breached": self.breached, "move": self.move.value,
                "reasons": self.reasons, "llr": self.llr,
                "monitor": self.monitor, "dissenter": self.dissenter,
                "skipped_rungs": self.skipped_rungs,
                "n_excluded": len(self.exclude_docs),
                "excluded": sorted(self.exclude_docs)}


def assess(turn: int, snap: Snapshot, band: Band, turns: list[SeatTurn],
           drift=None, rungs_used: int = 0, min_claims: int = 4,
           monitor: Sequential | None = None,
           self_anchor: dict[str, float] | None = None,
           territories: dict[str, str] | None = None) -> Assessment:
    """Judge one turn against the band and choose the next move.

    `rungs_used` is how many interventions this run has already spent, so the
    ladder is climbed rather than one rung pulled repeatedly: a rung that did
    not work is not more likely to work the second time.
    """
    monitor = monitor or Sequential()
    a = Assessment(turn=turn, verdict=Verdict.UNDETERMINED)

    if snap.n_claims < min_claims or snap.n_seats < 2:
        a.reasons.append(
            f"only {snap.n_claims} claim(s) from {snap.n_seats} seat(s); too "
            f"little to measure heterogeneity. This is NOT a healthy verdict.")
        a.llr, a.monitor = monitor.llr, monitor.to_json()
        return a

    if snap.participation < MIN_PARTICIPATION:
        a.verdict = Verdict.UNDETERMINED
        a.reasons.append(
            f"only {snap.n_seats} of {snap.n_expected} invited seats "
            f"contributed ({snap.participation:.0%}). Diversity among the "
            f"survivors of a panel that fell silent is not the diversity of the "
            f"panel: dropping the seats that agree with each other RAISES every "
            f"statistic here. Dispatch the missing packets, or re-run with the "
            f"roster you actually have -- which re-fixes the band.")
        a.llr, a.monitor = monitor.llr, monitor.to_json()
        return a

    st_declared = snap.n_stances > 0
    if turn == band.reference_turn:
        # The reference turn defines the band and cannot breach it. Judging it
        # against itself would be circular and would report the isolated first
        # round as COLLAPSING.
        a.verdict = Verdict.REFERENCE
        a.llr, a.monitor = monitor.llr, monitor.to_json()
        pos0 = snap.effective_positions
        band_abs_pos, abs_ok = abs_positions_floor(snap.backend)
        a.reasons.append(
            f"this is the isolated reference turn: it FIXES the band and is not "
            f"judged against it. VS2 {snap.vendi2:.2f} of {snap.vendi_max} "
            f"effective positions; separation evidence {snap.s('evidence'):.2f}, "
            f"positions {pos0:.2f}.")
        if not abs_ok:
            a.reasons.append(
                f"the degenerate-panel check was NOT MADE. Its floor is "
                f"calibrated on {CALIBRATED_BACKEND} and this turn was measured "
                f"with {snap.backend!r}, whose similarity scale is different; "
                f"the floors on record do not transfer, and applying them here "
                f"would manufacture a breach rather than detect one. Re-run on "
                f"{CALIBRATED_BACKEND} if you need this check, or calibrate "
                f"this backend and add it to ABS_FLOORS.")
        elif pos0 < band_abs_pos:
            a.breached.append(
                f"positions {pos0:.3f} < absolute floor {band_abs_pos:.3f} "
                f"({snap.backend})")
            a.reasons.append(
                "WARNING: this panel is already below the ABSOLUTE floor before "
                "any seat has seen any other. That is not collapse -- nothing "
                "has collapsed yet -- it is a degenerate panel: these seats do "
                "not produce distinct positions on this question even in "
                "isolation. Re-read the preflight, check whether the right "
                "seats were invited, and treat any later agreement as "
                "uninformative. A relative band built on a degenerate reference "
                "will certify the degeneracy.")
        return a

    ev, lx, pos = snap.s("evidence"), snap.s("lexical"), snap.effective_positions
    st = snap.s("stance")
    ev_ok = snap.informative("evidence")
    pos_ok = snap.informative("positions")

    if not ev_ok:
        a.reasons.append(
            "the evidence axis is UNINFORMATIVE this turn: its permutation null "
            "saturated, which happens when the seats hold disjoint corpora and "
            "re-dealing their citations still produces disjoint sets. It is "
            "excluded from the verdict rather than read as zero separation -- "
            "counting it would flag the healthiest possible configuration as a "
            "collapse. NOTE the cost: this is also the case where collapse "
            "would be hardest to see on this axis.")

    # The statistic the sequential monitor consumes: the share of TURN 1's
    # separation still present, per axis, taking the WORST axis rather than the
    # mean. Two reasons:
    #   * dividing by the FLOOR instead of by the turn-1 value silently reports
    #     something else entirely whenever an absolute floor binds
    #   * averaging a fully collapsed axis with a fully retained one gives 0.5,
    #     at which the Beta increment is exactly zero -- so a completely
    #     collapsed axis could contribute no evidence, forever
    retained: list[float] = []
    if ev_ok and band.ref_evidence > 0:
        retained.append(min(1.0, ev / band.ref_evidence))
    if pos_ok and band.ref_positions > 0:
        retained.append(min(1.0, pos / band.ref_positions))
    fired = monitor.update(min(retained) if retained else None)
    a.llr, a.monitor = monitor.llr, monitor.to_json()
    if not retained:
        a.reasons.append(
            "no axis could be measured against the reference this turn, so the "
            "sequential monitor received a MISSING observation and did not "
            "move. Missing is not health.")

    if ev_ok and ev < band.evidence:
        a.breached.append(f"evidence {ev:.3f} < {band.evidence:.3f}")
    if pos_ok and pos < band.positions:
        a.breached.append(f"positions {pos:.3f} < {band.positions:.3f}")
    if drift is not None and drift.novelty < NOVELTY_FLOOR:
        a.breached.append(
            f"novelty {drift.novelty:.2f} < {NOVELTY_FLOOR:.2f} -- this turn "
            f"restated the table rather than adding to it")
    # The lexical axis is REPORTED and never breaches. Form-based diversity
    # metrics score sets of random sentences as diverse (arXiv:2506.00514), so a
    # lexical floor would fire on the wrong thing and, worse, could be satisfied
    # by paraphrase. Appending it to `breached` would contradict this module's
    # own documentation.
    a.reasons.append(f"lexical separation {lx:.2f} (style channel; reported, "
                     f"never a floor).")

    # -- fidelity is a HARD constraint on the loop, not a footnote ---------
    if self_anchor and territories:
        lost = sorted(s for s, r in self_anchor.items()
                      if territories.get(s) == "home" and r < FIDELITY_FLOOR)
        if lost:
            a.verdict = Verdict.FIDELITY_LOST
            a.reasons.append(
                f"FIDELITY LOST: {', '.join(lost)} answered on their own "
                f"territory without anchoring in their own work. These seats "
                f"are competent generalists wearing a name, so NO verdict about "
                f"this turn's heterogeneity is meaningful -- a diversity gain "
                f"bought at this price is not a gain, and an agreement between "
                f"generalists is not an agreement between these scientists. "
                f"Fix this before reading any other number. No intervention is "
                f"applied, because the ladder cannot repair identity.")
            return a

    # -- the upper edge of the band ---------------------------------------
    if pos_ok and pos > band.ceiling_positions:
        a.verdict = Verdict.SCATTERED
        a.reasons.append(
            f"positions {pos:.2f} is ABOVE the ceiling "
            f"{band.ceiling_positions:.2f}: the table is more scattered than "
            f"the same seats were when isolated. That is not a better result. "
            f"It is what manufactured dissent and identity drift look like. "
            f"Check whether a prompt-perturbing intervention is in effect, and "
            f"check per-seat fidelity, before reading any of this turn's "
            f"disagreement as real.")
        return a

    # -- the disambiguating cases, before any verdict of collapse ---------
    #
    # HONEST SCOPE, AND IT IS A REAL LIMIT. These two branches read
    # DOCUMENT-IDENTITY overlap, not evidential independence. Five reviews of one
    # primary result count here as five different documents; two seats citing
    # the one decisive paper count as the same evidence. So the labels below
    # mean "agreement with disjoint citations" and "disagreement over shared
    # citations" -- which is weaker than independent corroboration and weaker
    # than a genuine argument, and is the right level of confidence to have from
    # a set operation over doc_ids. Resolving citations to underlying evidence
    # units (experiment, cohort, trial, primary result) is the top open problem
    # for this module; see docs/06-measurement.md.
    if ev_ok and ev >= band.evidence and st < 0.20 and st_declared:
        a.verdict = Verdict.CORROBORATION
        a.reasons.append(
            f"the seats agree ({1 - st:.0%} of declared stances) while citing "
            f"largely disjoint documents (separation {ev:.2f}, floor "
            f"{band.evidence:.2f}). Read as corroboration only if those "
            f"documents are not all reporting the same underlying result -- the "
            f"check here is document identity, not evidential independence. No "
            f"intervention: forcing dissent here would manufacture it.")
        return a

    if ev_ok and ev < band.evidence and st >= 0.35:
        a.verdict = Verdict.CONTESTED_EVIDENCE
        a.reasons.append(
            f"the seats are citing overlapping documents (evidence separation "
            f"{ev:.2f}) and disagreeing ({st:.0%} dissent, stance entropy "
            f"{snap.stance_entropy:.2f}). They are arguing over shared text, "
            f"which is what a round table is for. No intervention.")
        return a

    if not a.breached:
        a.verdict = Verdict.HEALTHY
        a.reasons.append(
            f"every axis is inside the band (evidence {ev:.2f}, lexical "
            f"{lx:.2f}, positions {pos:.2f}, VS2 {snap.vendi2:.2f} of "
            f"{snap.vendi_max}).")
        return a

    if fired != "collapsing":
        # Breached, but the monitor has not accumulated enough evidence to act.
        # Reported rather than acted on: intervening on one noisy turn is how a
        # guardrail becomes the thing that caused the result.
        a.verdict = Verdict.COLLAPSING
        a.reasons.append(
            f"axes below the band this turn, but the sequential monitor has not "
            f"fired (LLR {monitor.llr:+.2f} of {monitor.upper:+.2f}). Watching. "
            f"One turn at this panel size is too noisy to act on.")
        return a

    if drift is not None and drift.herding > 0.5:
        a.reasons.append(
            f"herding {drift.herding:+.2f}: seats that changed a stance this "
            f"turn moved toward the majority, not away from it")

    a.verdict = Verdict.COLLAPSING
    a.reasons.append(
        "heterogeneity has fallen below what this panel produced when the seats "
        "could not see each other, and has stayed there long enough for the "
        "sequential monitor to fire. On this question, that is a property of "
        "the conversation rather than of the science.")

    rung = LADDER[min(rungs_used, len(LADDER) - 1)]
    a.move = rung
    if rung is Move.DIFFERENTIAL_RETRIEVAL:
        a.exclude_docs = consensus_docs(turns, 0.5)
        if not a.exclude_docs:
            # Nothing shared to withhold: the convergence is not arriving
            # through the evidence, so skip a rung that cannot bite. `skipped`
            # is reported so the caller advances the ladder by TWO -- otherwise
            # the skipped rung is re-selected on the next firing and the ladder
            # never moves.
            a.move = LADDER[min(rungs_used + 1, len(LADDER) - 1)]
            a.skipped_rungs = 1
            a.reasons.append(
                "no document is cited by half the table, so withholding shared "
                "evidence cannot bite; that rung is skipped, not retried.")
    if a.move is Move.ASSIGN_DISSENT:
        a.dissenter = _evidenced_minority(turns)
        if a.dissenter:
            a.reasons.append(
                f"dissent assigned to `{a.dissenter}`: of the seats that "
                f"disagreed with someone this turn, it stands on the most "
                f"evidence the rest of the table is not using. Assigning it to "
                f"an ARBITRARY seat is the version that degrades identity.")
        else:
            a.reasons.append(
                "no seat holds an evidenced minority position, so there is "
                "nobody to assign dissent to without inventing it. Skipping to "
                "the next rung rather than manufacturing a dissenter.")
            a.move = LADDER[min(rungs_used + 1, len(LADDER) - 1)]
            a.skipped_rungs = 1
    if a.move in (Move.TERMINATE, Move.DECLARE):
        # Both end the run. TERMINATE says "stop deliberating"; the run is over
        # either way, so the verdict is COLLAPSED and DECLARE is not left
        # stranded behind a rung that refuses the next turn.
        a.verdict = Verdict.COLLAPSED
        a.reasons.append(
            "the ladder is exhausted. Read the transcript as the output of a "
            "collapsed panel: its agreement carries no more weight than one "
            "agent's, and no further turn will change that.")
    return a


def _evidenced_minority(turns: list[SeatTurn]) -> str:
    """The seat that disagrees AND stands on evidence the others are not using.

    Assigning dissent to an arbitrary devil's advocate is the version that
    backfires: debate-framing instructions measurably increase persona
    inconstancy (arXiv:2405.03862). Assigning it to the seat whose own evidence
    already supports a minority position asks for something it can actually do.

    Returns "" when nobody qualifies, which is the honest answer and is treated
    as a reason to skip the rung rather than to pick someone.
    """
    dissenting = [t for t in turns
                  if any(v not in ("agree", "out_of_scope")
                         for v in t.stances.values())]
    if not dissenting:
        return ""
    best, best_score = "", -1
    for t in dissenting:
        others = {d for o in turns if o.seat != t.seat for d in o.cited_docs}
        unique = len(t.cited_docs - others)
        if unique > best_score:
            best, best_score = t.seat, unique
    return best if best_score > 0 else ""


NUDGE = """\
A HETEROGENEITY INTERVENTION IS IN EFFECT FOR THIS TURN.

Measured across the whole table, the difference between the seats has fallen
below what this same panel produced on this same question when no seat could see
any other, and it has stayed there long enough that it is not noise. That is a
property of the conversation, not of the science.

{description}

This is NOT an instruction to disagree. Manufactured disagreement is worse than
agreement, because it is invisible in the transcript -- and instructions to
argue measurably make an agent LESS consistent with the identity it is supposed
to hold. It is an instruction to answer from where you actually stand: your own
results first, your own methods, the objections your own work commits you to.
If, having done that, you still agree with the others, say so plainly and say
which of your own evidence takes you there. That is a real finding and it will
be recorded as one.
"""


def nudge_block(move: Move) -> str:
    """The ONLY rung that changes what a seat is told.

    Every other rung changes what a seat can READ or SEE. Attaching this block
    to all of them would make a recovery after withholding shared evidence
    uninterpretable, because the seats would also have been told the table was
    converging. The routing and evidence rungs are recorded in the transcript
    and nowhere else.
    """
    return NUDGE.format(description=MOVE_DESCRIPTION[move]) \
        if move is Move.ASSIGN_DISSENT else ""
