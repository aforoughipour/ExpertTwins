"""How much heterogeneity is left in the conversation, measured five ways.

WHY FIVE AND NOT ONE.

A single diversity number is gameable and, worse, ambiguous. Seats that all say
different words about the same evidence look diverse lexically and are not.
Seats that reach the same conclusion from disjoint evidence look collapsed on a
verdict metric and are in fact the strongest possible result -- independent
corroboration. So the axes are kept separate and the verdict is a function of
the PATTERN across them, never of a scalar.

    evidence      whose paper is each seat standing on
    lexical       whose words is it using -- A STYLE CHANNEL ONLY, never a floor
    positions     how many genuinely distinct positions are on the table (VS-2)
    stance        who is contradicting whom, declared explicitly
    herding       who moved toward the majority between turns (see `drift`)

CHANCE CORRECTION, AND WHY IT IS THE WHOLE TRICK -- AND WHERE IT DOES NOT APPLY.

Raw pairwise distance is uninterpretable: 0.6 is high for seats sharing one
corpus and low for seats with disjoint ones. The SET-BASED axes -- evidence and
lexical -- are therefore reported as a CHANCE-CORRECTED SEPARATION

    S = (H_observed - H_null) / (1 - H_null)

where H_null is the same statistic recomputed after the claims have been
randomly re-dealt among the seats. H_null is, operationally, *what this panel
would look like if it were one agent talking to itself*: shuffling destroys the
seat->claim association and leaves only the pooled distribution.

This is exactly the form of chance-corrected agreement in Cohen's kappa
(doi:10.1177/001316446002000104), Fleiss's kappa (doi:10.1037/h0031619) and
Krippendorff's alpha (doi:10.1093/hcr/30.3.411, doi:10.1080/19312450709336664) --
(observed - expected) / (maximum - expected) -- with the expected term estimated
by permutation rather than assumed. It is the reason a threshold can be stated
at all: S = 0 means indistinguishable from a single agent, S = 1 means maximally
separated, and both ends mean the same thing regardless of topic or corpus.

TWO PLACES THE CORRECTION FAILS, BOTH FOUND BY TESTING RATHER THAN BY THOUGHT:

  * When seats hold DISJOINT corpora -- which is the design -- re-dealing their
    citations still yields disjoint sets, so the Jaccard evidence statistic pins
    at 1.0 under both the observation and the null. The correction then reports
    the healthiest possible configuration as zero separation. Such an axis is
    marked `informative = False` and the guardrail EXCLUDES it rather than
    reading it as collapse.
  * The POSITIONS axis is not chance-corrected at all. VS-2 is already on an
    absolute, interpretable scale -- an effective number of distinct positions --
    and re-dealing mutually distinct claims leaves it almost unchanged, so the
    null saturates for the same structural reason. Its chance-correction happens
    one level up, in `guardrail.py`, where the band is anchored on the blind
    first turn. The permutation null is still computed and reported, as a
    diagnostic.

TWO METRIC CHOICES HERE ARE NOT PREFERENCES BUT FINITE-SAMPLE RESULTS.

  * The semantic axis uses the ORDER-2 Vendi Score (RKE), not the plain one. The
    order-1 score does not reliably converge below about 20,000 samples, while
    the order-2 score has universal convergence guarantees and the t-truncated
    score converges at n = O(t) (Ospanov & Farnia, arXiv:2410.21719). This panel
    has ten to fifteen seats, so order 1 is the wrong estimator here.
  * The lexical axis is reported and is NOT allowed to be a floor. Form-based
    diversity metrics assign high diversity to sets of random sentences
    (arXiv:2506.00514) and distinct-n is length-biased (arXiv:2202.13587).

Every citation in this module is recorded with its verification status in
`docs/06-measurement.md`. Do not quote one from here into a paper
without re-fetching it.

    HONEST STATUS. The individual metrics are standard and cited. The
    PERMUTATION NULL applied to an embedding statistic, the composite verdict in
    `guardrail.py`, and every constant are EXTRAPOLATION -- reasoning, not a
    published result. A dedicated search found no published adaptive diversity
    FLOOR calibrated to topic contestedness for LLM round tables; the nearest
    prior art is a round-count governor (arXiv:2605.19193) and diversity-
    preserving peer routing (arXiv:2609.00683), neither of which is this.
    Nothing here has been validated against a labelled collapse. Read the
    pattern, not the digits.

NO GPU, NO NETWORK, NO MODEL by default. `numpy` is the only hard dependency.
An embedding model is used for the semantic axis IF one is importable, and a
tf-idf cosine kernel is used when it is not; which one ran is recorded in the
output, because a metric whose backend is unknown is not a measurement.
"""
from __future__ import annotations

import math
import os
import random
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Iterable, Sequence

import numpy as np

_WORD = re.compile(r"[A-Za-z][A-Za-z0-9\-]+")

_STOP = {
    "the", "and", "for", "that", "this", "with", "from", "are", "was", "were",
    "not", "but", "which", "these", "those", "have", "has", "had", "can", "may",
    "will", "would", "could", "should", "than", "then", "there", "their", "its",
    "our", "your", "his", "her", "they", "them", "what", "when", "how", "why",
    "who", "into", "over", "under", "between", "within", "without", "also",
    "been", "being", "such", "more", "most", "other", "same", "does", "did",
    "any", "all", "each", "both", "some", "one", "two", "three",
}

#: Markers of live scientific argument. Used ONLY to estimate how contested a
#: topic is, never to score a claim. Crude by design and reported as crude.
_CONTEST_MARKERS = (
    "however", "in contrast", "by contrast", "conversely", "nevertheless",
    "controversial", "controversy", "debate", "debated", "disputed",
    "conflicting", "inconsistent", "discrepant", "discrepancy", "paradox",
    "remains unclear", "remains unknown", "not been established",
    "contrary to", "challenge the", "challenges the", "fail to replicate",
    "could not be replicated", "unresolved", "at odds with",
)


def tokens(text: str) -> list[str]:
    return [t.lower() for t in _WORD.findall(text or "")
            if len(t) > 2 and t.lower() not in _STOP]


# --------------------------------------------------------------------------
# primitive distances
# --------------------------------------------------------------------------

def jaccard_distance(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    u = len(a | b)
    return 1.0 - (len(a & b) / u) if u else 0.0


def js_divergence(p: Counter, q: Counter, base: float = 2.0) -> float:
    """Jensen-Shannon divergence between two term distributions.

    Bounded in [0, 1] with base 2, symmetric, finite even when the supports are
    disjoint -- which is why it is used here rather than KL: seats routinely use
    vocabularies with no overlap at all, and KL would be infinite.
    """
    if not p or not q:
        return 0.0 if (not p and not q) else 1.0
    np_, nq = sum(p.values()), sum(q.values())
    keys = set(p) | set(q)
    out = 0.0
    for k in keys:
        pi = p.get(k, 0) / np_
        qi = q.get(k, 0) / nq
        mi = 0.5 * (pi + qi)
        if pi > 0:
            out += 0.5 * pi * math.log(pi / mi, base)
        if qi > 0:
            out += 0.5 * qi * math.log(qi / mi, base)
    return max(0.0, min(1.0, out))


def distinct_n(texts: Iterable[str], n: int = 2) -> float:
    """Fraction of n-grams that are unique across a pooled set of texts."""
    grams: list[tuple[str, ...]] = []
    for t in texts:
        tk = tokens(t)
        grams += [tuple(tk[i:i + n]) for i in range(max(0, len(tk) - n + 1))]
    return (len(set(grams)) / len(grams)) if grams else 0.0


# --------------------------------------------------------------------------
# semantic axis
# --------------------------------------------------------------------------

def tfidf_matrix(texts: Sequence[str]) -> np.ndarray:
    """L2-normalised tf-idf rows. Pure numpy, no sklearn, no network."""
    docs = [tokens(t) for t in texts]
    vocab = sorted({w for d in docs for w in d})
    if not vocab or not docs:
        return np.zeros((len(texts), 1))
    idx = {w: i for i, w in enumerate(vocab)}
    tf = np.zeros((len(docs), len(vocab)))
    for r, d in enumerate(docs):
        for w, c in Counter(d).items():
            tf[r, idx[w]] = 1.0 + math.log(c)
    df = (tf > 0).sum(axis=0)
    idf = np.log((1 + len(docs)) / (1 + df)) + 1.0
    m = tf * idf
    norm = np.linalg.norm(m, axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    return m / norm


#: Backends whose absolute floors have been calibrated. A similarity number is
#: meaningful only against the instrument that produced it, so the name of that
#: instrument is the first thing every consumer of `embed` is handed.
CALIBRATED_BACKEND = "tfidf-cosine"

#: Set by `embed` when the preferred encoder could not be used. Kept so a caller
#: can report the reason rather than only the consequence.
LAST_FALLBACK_REASON = ""


@lru_cache(maxsize=4)
def _encoder(model_name: str):                              # pragma: no cover
    """Load a sentence-transformers encoder ONCE per process.

    Constructing `SentenceTransformer` reads the weights from disk and builds
    the torch graph; it takes seconds. `embed` is called once per axis per turn
    and many times over in the test suite, so constructing it per call made a
    45-second run take over ten minutes. Caching is safe because the object is
    used read-only, and it is keyed on the model name so switching backends
    mid-process still loads the right weights.
    """
    from sentence_transformers import SentenceTransformer  # type: ignore
    return SentenceTransformer(model_name)


def embed(texts: Sequence[str]) -> tuple[np.ndarray, str]:
    """Embed texts, preferring a real encoder and degrading loudly, not silently.

    Returns (unit-norm rows, backend name). The backend name is carried into
    every report: a similarity number whose backend is unrecorded cannot be
    compared against another run, and comparing across runs is the entire use of
    this axis.

    THE FALLBACK IS ANNOUNCED ON STDERR, AND THAT IS NOT COSMETIC. Silently
    swallowing encoder failures and returning a tf-idf matrix would directly
    contradict this docstring. The two backends do not merely differ in
    precision -- they differ in SCALE. A run that silently changed backend would
    change the meaning of every threshold in `guardrail.py` without changing a
    single number on screen.
    """
    global LAST_FALLBACK_REASON
    LAST_FALLBACK_REASON = ""
    model_name = os.environ.get("EXPERTWINS_EMBED_MODEL", "all-MiniLM-L6-v2")
    try:                                                    # pragma: no cover
        from sentence_transformers import SentenceTransformer  # type: ignore
    except Exception as exc:                                # noqa: BLE001
        # The documented no-model path: numpy is the only hard dependency and
        # this is the configuration the floors were calibrated on. Stated once,
        # quietly, because it is expected rather than wrong.
        LAST_FALLBACK_REASON = f"sentence-transformers not importable ({exc})"
        return tfidf_matrix(texts), CALIBRATED_BACKEND
    try:                                                    # pragma: no cover
        m = _encoder(model_name)
        v = np.asarray(m.encode(list(texts), normalize_embeddings=True))
        return v, f"sentence-transformers:{model_name}"
    except Exception as exc:                                # noqa: BLE001
        # The encoder is INSTALLED and still did not run: a bad model name, a
        # failed download, an offline cache miss. That is a different event from
        # the one above and it is never expected, so it is loud.
        LAST_FALLBACK_REASON = f"{model_name!r} failed to load ({exc})"
        print(
            f"\n  EMBEDDING BACKEND FELL BACK TO {CALIBRATED_BACKEND}.\n"
            f"  sentence-transformers is installed but {model_name!r} did not "
            f"load:\n    {exc}\n"
            f"  This CHANGES THE SCALE of every semantic number below, not just "
            f"their precision.\n"
            f"  Numbers from this run are comparable only with other "
            f"{CALIBRATED_BACKEND} runs.",
            file=sys.stderr)
        return tfidf_matrix(texts), CALIBRATED_BACKEND


def _kernel_eigenvalues(vectors: np.ndarray) -> np.ndarray:
    """Normalised eigenvalues of the similarity kernel of unit vectors."""
    n = vectors.shape[0]
    if n == 0:
        return np.zeros(0)
    k = vectors @ vectors.T
    k = np.clip((k + k.T) / 2.0, -1.0, 1.0)
    w = np.linalg.eigvalsh(k / n)
    w = w[w > 1e-12]
    return w / w.sum() if w.size else w


def vendi_order2(vectors: np.ndarray) -> float:
    """VS-2 / RKE: the order-2 Vendi Score, ``1 / sum(lambda_i^2)``.

    THIS, AND NOT THE PLAIN VENDI SCORE, IS THE ONE TO USE HERE, AND THE REASON
    IS A FINITE-SAMPLE RESULT RATHER THAN A PREFERENCE.

    The original Vendi Score (Friedman & Dieng, arXiv:2210.02410) is the
    exponentiated matrix-based Shannon entropy of the kernel eigenvalues -- the
    order-1 member of a family of kernel entropic diversity scores
    (arXiv:2310.12952). Ospanov & Farnia (arXiv:2410.21719) show that the order-1
    score computed at sample sizes below roughly 20,000 may not converge to its
    asymptotic value, while the order-2 score (RKE) "enjoys universal
    convergence guarantees across all kernel functions".

    This panel has ten to fifteen seats. Order 1 is therefore not merely noisy
    here, it is the wrong estimator.
    All three citations are VERIFIED in `docs/06-measurement.md`.

    Reads as an EFFECTIVE NUMBER OF DISTINCT POSITIONS, in [1, n]. With ten
    seats: 6-8 is healthy heterogeneity, 2-3 means two camps have formed, and
    approaching 1.5 is collapse.
    """
    w = _kernel_eigenvalues(vectors)
    if w.size == 0:
        return 0.0
    return float(1.0 / float((w ** 2).sum()))


def vendi_truncated(vectors: np.ndarray, t: int = 4) -> float:
    """The t-truncated Vendi Score: order-1 entropy over the top-t eigenvalues.

    Provably converges at n = O(t) samples (arXiv:2410.21719), so it is safe at
    this panel size where the untruncated order-1 score is not. Reported
    alongside VS-2 as a second view of the same kernel; `t` must be held fixed
    across rounds or the numbers are not comparable.
    """
    w = _kernel_eigenvalues(vectors)
    if w.size == 0:
        return 0.0
    top = np.sort(w)[::-1][:max(1, t)]
    top = top / top.sum()
    return float(math.exp(-float((top * np.log(top)).sum())))


def cosine_dispersion(vectors: np.ndarray) -> float:
    """1 - mean pairwise cosine. Cheap and monotone; a plotting statistic only.

    Deliberately NOT used as a floor: it cannot tell "spread out" from "two tight
    camps", which is exactly the state a collapsing panel passes through, and
    which VS-2 does distinguish (two camps drive VS-2 toward 2).
    """
    n = vectors.shape[0]
    if n < 2:
        return 0.0
    k = vectors @ vectors.T
    iu = np.triu_indices(n, 1)
    return float(1.0 - k[iu].mean())


def stance_entropy(counts: Counter, n_categories: int = 4) -> float:
    """Normalised Shannon entropy of the stance distribution, bias-corrected.

    The plug-in entropy estimator is downward-biased at small n -- and n here is
    the number of declared stances at one table, which is small. The
    Miller-Madow correction adds (K_observed - 1) / (2n) before normalising.

    Entropy over stance CATEGORIES rather than over surface strings is the
    construction Farquhar et al. use for semantic entropy (Nature 2024,
    doi:10.1038/s41586-024-07421-0): cluster into equivalence classes first,
    count second. Here the clustering is free, because seats declare a stance
    from a closed enum rather than writing prose that has to be classified.
    """
    n = sum(counts.values())
    if n == 0 or n_categories < 2:
        return 0.0
    h = 0.0
    for c in counts.values():
        p = c / n
        if p > 0:
            h -= p * math.log(p)
    h += (len([c for c in counts.values() if c > 0]) - 1) / (2.0 * n)
    return max(0.0, min(1.0, h / math.log(n_categories)))


def modal_disagreement(counts: Counter) -> float:
    """1 - (modal share). The bluntest and most robust dissent statistic."""
    n = sum(counts.values())
    if n == 0:
        return 0.0
    return float(1.0 - max(counts.values()) / n)


# --------------------------------------------------------------------------
# the per-turn observation
# --------------------------------------------------------------------------

@dataclass
class SeatTurn:
    """What one seat contributed in one turn, reduced to what diversity needs."""

    seat: str
    claim_texts: list[str] = field(default_factory=list)
    cited_docs: set[str] = field(default_factory=set)
    #: seat -> "agree" | "disagree" | "out_of_scope" | "needs_other_evidence"
    stances: dict[str, str] = field(default_factory=dict)


@dataclass
class Axis:
    name: str
    observed: float
    null: float
    separation: float
    detail: str = ""
    #: False when the statistic cannot discriminate here -- typically because
    #: the permutation null has saturated. An axis that cannot discriminate must
    #: be EXCLUDED from the verdict, not counted as a breach: reporting
    #: "separation 0.00" for seats whose evidence is disjoint by construction
    #: would flag the healthiest possible configuration as a collapse.
    informative: bool = True

    def render(self) -> str:
        flag = "" if self.informative else "  [UNINFORMATIVE -- null saturated]"
        return (f"{self.name:<12s} obs {self.observed:6.3f}  null {self.null:6.3f}  "
                f"S {self.separation:6.3f}  {self.detail}{flag}")


@dataclass
class Snapshot:
    turn: int
    axes: dict[str, Axis]
    vendi2: float
    vendi_t: float
    cos_dispersion: float
    stance_entropy: float
    modal_disagreement: float
    n_stances: int
    vendi_max: int
    backend: str
    n_seats: int
    n_expected: int
    n_claims: int

    def s(self, name: str) -> float:
        a = self.axes.get(name)
        return a.separation if a else 0.0

    def informative(self, name: str) -> bool:
        a = self.axes.get(name)
        return bool(a and a.informative)

    @property
    def effective_positions(self) -> float:
        """Chance-corrected separation on the semantic axis."""
        return self.s("positions")

    def render(self) -> str:
        head = f"turn {self.turn}: {self.n_seats} seats"
        if self.n_expected > self.n_seats:
            head += (f" of {self.n_expected} expected -- "
                     f"{self.n_expected - self.n_seats} SILENT")
        lines = [f"{head}, {self.n_claims} claims, {self.n_stances} stances"]
        for a in self.axes.values():
            lines.append("  " + a.render())
        lines.append(
            f"  {'raw':<12s} VS2 {self.vendi2:5.2f} of {self.vendi_max}   "
            f"VS_t4 {self.vendi_t:5.2f}   cos-disp {self.cos_dispersion:5.3f}   "
            f"stance-H {self.stance_entropy:4.2f}   modal-dis "
            f"{self.modal_disagreement:4.2f}   [{self.backend}]")
        return "\n".join(lines)

    @property
    def participation(self) -> float:
        """Share of the invited roster that actually contributed.

        Silent seats are NOT free. `measure` drops a seat with nothing to say,
        so a ten-seat panel reduced to its two most different members would
        otherwise score near-maximal normalised diversity. The denominator is
        kept so that cannot happen quietly.
        """
        return (self.n_seats / self.n_expected) if self.n_expected else 0.0

    def to_json(self) -> dict:
        return {
            "turn": self.turn, "n_seats": self.n_seats,
            "n_expected": self.n_expected, "participation": self.participation,
            "n_claims": self.n_claims, "n_stances": self.n_stances,
            "backend": self.backend, "vendi2": self.vendi2,
            "vendi_truncated_t4": self.vendi_t,
            "cos_dispersion": self.cos_dispersion,
            "stance_entropy": self.stance_entropy,
            "modal_disagreement": self.modal_disagreement,
            "vendi_max": self.vendi_max,
            "effective_positions": self.effective_positions,
            "axes": {k: {"observed": a.observed, "null": a.null,
                         "separation": a.separation, "detail": a.detail,
                         "informative": a.informative}
                     for k, a in self.axes.items()},
        }


#: A permutation null this close to the ceiling leaves no room for the observed
#: statistic to exceed it, so the chance-corrected separation is not small --
#: it is undefined. Saturation happens for a legitimate reason here: when seats
#: hold disjoint corpora, re-dealing their citations still produces disjoint
#: sets, so a Jaccard-based evidence statistic pins at 1.0 whichever way the
#: claims are shuffled. An axis in that state is marked uninformative and the
#: guardrail ignores it, rather than reading it as collapse.
NULL_SATURATION = 0.95


def _informative(observed: float, null: float) -> bool:
    return null < NULL_SATURATION


def _mean_pairwise(items: Sequence, fn) -> float:
    vals = [fn(items[i], items[j])
            for i in range(len(items)) for j in range(i + 1, len(items))]
    return float(sum(vals) / len(vals)) if vals else 0.0


def _separation(observed: float, null: float) -> float:
    """(observed - null) / (1 - null), clipped. See the module docstring."""
    if null >= 1.0:
        return 0.0
    return max(0.0, min(1.0, (observed - null) / (1.0 - null)))


def _shuffle_null(turns: Sequence[SeatTurn], fn, n_perm: int = 200,
                  seed: int = 20260910) -> float:
    """Re-deal every claim among the seats and recompute. Sizes are preserved.

    Preserving each seat's claim count matters: a seat that filed nine claims
    and one that filed one are not interchangeable, and a null that ignores that
    would make an unbalanced panel look separated when it is merely lopsided.
    """
    pool: list[tuple[str, set[str]]] = []
    sizes: list[int] = []
    for t in turns:
        sizes.append(len(t.claim_texts))
        for i, text in enumerate(t.claim_texts):
            pool.append((text, t.cited_docs if i == 0 else set()))
    # Documents are re-dealt independently of claims, because a seat's evidence
    # SET is the unit of interest on the evidence axis. The pool is DEDUPLICATED
    # per seat but not globally, so a document cited by three seats appears three
    # times and can land in one fake seat twice; the slice is then set()-ed and
    # that seat comes out smaller than the seat it stands in for. Sizes are
    # therefore restored by topping the slice up from the remaining pool.
    doc_pool: list[str] = [d for t in turns for d in t.cited_docs]
    doc_sizes = [len(t.cited_docs) for t in turns]
    if not pool:
        return 0.0

    rng = random.Random(seed)
    vals: list[float] = []
    for _ in range(n_perm):
        texts = [p[0] for p in pool]
        rng.shuffle(texts)
        rng.shuffle(doc_pool)
        fake: list[SeatTurn] = []
        ti = 0
        cursor = 0
        for t, n, dn in zip(turns, sizes, doc_sizes):
            docs: set[str] = set()
            while len(docs) < dn and cursor < len(doc_pool):
                docs.add(doc_pool[cursor])
                cursor += 1
            fake.append(SeatTurn(
                seat=t.seat,
                claim_texts=texts[ti:ti + n],
                cited_docs=docs,
                stances=t.stances))
            ti += n
        vals.append(fn(fake))
    return float(sum(vals) / len(vals)) if vals else 0.0


# -- the individual axis statistics ---------------------------------------

def _evidence_obs(turns: Sequence[SeatTurn]) -> float:
    return _mean_pairwise([t.cited_docs for t in turns], jaccard_distance)


def _lexical_obs(turns: Sequence[SeatTurn]) -> float:
    counters = [Counter(tokens(" ".join(t.claim_texts))) for t in turns]
    return _mean_pairwise(counters, js_divergence)


def _stance_obs(turns: Sequence[SeatTurn]) -> float:
    """Fraction of declared stances that are not agreement.

    `out_of_scope` counts as neither: a seat declining to rule on something
    outside its territory is being faithful, not disagreeing, and counting it as
    dissent would reward abstention as though it were an argument.
    """
    total = dissent = 0
    for t in turns:
        for _, v in t.stances.items():
            if v == "out_of_scope":
                continue
            total += 1
            if v != "agree":
                dissent += 1
    return (dissent / total) if total else 0.0


def _stance_counts(turns: Sequence[SeatTurn]) -> Counter:
    c: Counter = Counter()
    for t in turns:
        for v in t.stances.values():
            c[v] += 1
    return c


def measure(turns: Sequence[SeatTurn], turn: int = 1,
            n_perm: int = 200, n_expected: int | None = None) -> Snapshot:
    """One turn's heterogeneity, on every axis, chance-corrected where that is
    meaningful.

    `n_expected` is the size of the INVITED roster. Seats with nothing to say
    are dropped from the statistics -- they cannot contribute a position -- but
    the denominator is kept, because otherwise a panel that has fallen silent
    scores as a panel that is highly diverse.
    """
    invited = n_expected if n_expected is not None else len(turns)
    turns = [t for t in turns if t.claim_texts or t.cited_docs]
    axes: dict[str, Axis] = {}

    ev = _evidence_obs(turns)
    ev_null = _shuffle_null(turns, _evidence_obs, n_perm)
    shared = _consensus_docs(turns, 0.5)
    axes["evidence"] = Axis(
        "evidence", ev, ev_null, _separation(ev, ev_null),
        f"{len(shared)} doc(s) cited by half the table or more",
        informative=_informative(ev, ev_null))

    lx = _lexical_obs(turns)
    lx_null = _shuffle_null(turns, _lexical_obs, n_perm)
    axes["lexical"] = Axis(
        "lexical", lx, lx_null, _separation(lx, lx_null),
        "STYLE CHANNEL ONLY -- form-based metrics score random sentences as "
        "diverse (arXiv:2506.00514); never read this as the floor",
        informative=_informative(lx, lx_null))

    st = _stance_obs(turns)
    n_declared = sum(len(t.stances) for t in turns)
    # No permutation null: shuffling WHO holds a stance does not change how many
    # stances are dissenting. The separation is therefore the raw rate, and that
    # is recorded rather than dressed up as chance-corrected. When NO stance was
    # declared the axis is UNINFORMATIVE -- zero dissent among zero declarations
    # is not agreement, and reading it as agreement is how a table that simply
    # stopped answering the stance question gets certified as converged.
    axes["stance"] = Axis(
        "stance", st, 0.0, st,
        f"{n_declared} declared stance(s); null is trivial"
        + ("" if n_declared else " -- NONE DECLARED, so this is not agreement"),
        informative=bool(n_declared))

    # -- the semantic axis -------------------------------------------------
    # NOT chance-corrected, and that is deliberate. VS-2 is already on an
    # absolute, interpretable scale -- an effective number of distinct positions
    # in [1, m] -- which means the same thing regardless of corpus or topic, and
    # that is the entire point of the Vendi family. The permutation null is
    # computed and REPORTED as a diagnostic, but it must not be subtracted:
    # re-dealing mutually distinct claims among seats leaves VS-2 almost
    # unchanged, so the null saturates and the correction would report a
    # perfectly separated panel as having zero separation. The chance-correction
    # this axis needs happens one level up, in `guardrail.py`, where the band is
    # anchored on the blind first turn.
    texts = [" ".join(t.claim_texts) for t in turns if t.claim_texts]
    m = len(texts)
    if m >= 2:
        vecs, backend = embed(texts)
        v2 = vendi_order2(vecs)
        vt = vendi_truncated(vecs, 4)
        cosd = cosine_dispersion(vecs)
        obs = (v2 - 1.0) / (m - 1.0)

        def _positions_obs(fake: Sequence[SeatTurn]) -> float:
            ft = [" ".join(t.claim_texts) for t in fake if t.claim_texts]
            if len(ft) < 2:
                return 0.0
            fv, _ = embed(ft)
            return (vendi_order2(fv) - 1.0) / (len(ft) - 1.0)

        # Embedding a permuted table is the expensive part, so this null uses
        # far fewer permutations than the set-based axes. It is a diagnostic,
        # not a correction.
        pos_null = _shuffle_null(turns, _positions_obs, max(8, n_perm // 25))
        axes["positions"] = Axis(
            "positions", obs, pos_null, obs,
            f"VS2 {v2:.2f} of {m} -- effective number of distinct positions; "
            f"absolute scale, NOT chance-corrected (null shown for diagnosis)")
    else:
        v2 = vt = cosd = 0.0
        backend = "none"
        axes["positions"] = Axis("positions", 0.0, 0.0, 0.0,
                                 "fewer than two seats filed grounded claims",
                                 informative=False)

    counts = _stance_counts(turns)
    return Snapshot(
        turn=turn, axes=axes, vendi2=v2, vendi_t=vt, cos_dispersion=cosd,
        stance_entropy=stance_entropy(counts, n_categories=4),
        modal_disagreement=modal_disagreement(counts),
        n_stances=sum(counts.values()),
        vendi_max=max(1, m), backend=backend, n_seats=len(turns),
        n_expected=max(invited, len(turns)),
        n_claims=sum(len(t.claim_texts) for t in turns))


def _consensus_docs(turns: Sequence[SeatTurn], frac: float = 0.5) -> set[str]:
    """Documents cited by at least `frac` of the seats that cited anything.

    The handle the guardrail pulls: these are the documents to withhold when the
    table converges, because shared evidence is the most direct mechanical
    cause of shared conclusions.
    """
    citing = [t for t in turns if t.cited_docs]
    if not citing:
        return set()
    c: Counter = Counter()
    for t in citing:
        c.update(t.cited_docs)
    need = max(2, math.ceil(frac * len(citing)))
    return {d for d, n in c.items() if n >= need}


def consensus_docs(turns: Sequence[SeatTurn], frac: float = 0.5) -> set[str]:
    return _consensus_docs(turns, frac)


# --------------------------------------------------------------------------
# between-turn dynamics
# --------------------------------------------------------------------------

@dataclass
class Drift:
    """What changed between two turns."""

    novelty: float          # share of claims not already on the table
    herding: float          # net movement toward the majority stance, in [-1, 1]
    delta_separation: dict[str, float]

    def render(self) -> str:
        d = "  ".join(f"{k} {v:+.3f}" for k, v in self.delta_separation.items())
        return (f"novelty {self.novelty:.2f}  herding {self.herding:+.2f}  "
                f"dS: {d}")


def drift(prev: Sequence[SeatTurn], cur: Sequence[SeatTurn],
          prev_snap: Snapshot, cur_snap: Snapshot,
          near_duplicate: float = 0.80) -> Drift:
    """Novelty and herding between consecutive turns.

    Novelty is the share of this turn's claims that are not near-duplicates of
    anything already said by ANY seat -- including by the speaker. A turn in
    which everyone restates the table's existing claims in fresh words scores
    high on lexical diversity and near zero here, which is exactly the failure
    lexical metrics miss.
    """
    old = [c for t in prev for c in t.claim_texts]
    new = [c for t in cur for c in t.claim_texts]
    if new and old:
        vecs, _ = embed(old + new)
        o, n = vecs[:len(old)], vecs[len(old):]
        sim = n @ o.T
        novelty = float((sim.max(axis=1) < near_duplicate).mean())
    else:
        novelty = 1.0 if new else 0.0

    before = {t.seat: t.stances for t in prev}
    toward = away = 0
    for t in cur:
        prior = before.get(t.seat, {})
        for other, now in t.stances.items():
            was = prior.get(other)
            if was is None or was == now:
                continue
            if now == "agree":
                toward += 1
            elif was == "agree":
                away += 1
    moved = toward + away
    herding = ((toward - away) / moved) if moved else 0.0

    return Drift(
        novelty=novelty,
        herding=herding,
        delta_separation={k: cur_snap.s(k) - prev_snap.s(k)
                          for k in cur_snap.axes},
    )


# --------------------------------------------------------------------------
# preflight: how much heterogeneity should we EXPECT here?
# --------------------------------------------------------------------------

@dataclass
class Preflight:
    """Computed from the packets alone, before a single token is generated."""

    packet_separation: float
    rubric_separation: float
    contestedness: float
    retention: float
    n_seats: int
    notes: list[str] = field(default_factory=list)

    def render(self) -> str:
        return "\n".join([
            f"packet separation   {self.packet_separation:.3f}   "
            f"(question-conditioned; how much do these seats even read in common?)",
            f"rubric separation   {self.rubric_separation:.3f}   "
            f"(do they refuse the same things?)",
            f"contestedness       {self.contestedness:.3f}   "
            f"(is the literature itself arguing?)",
            f"retention rho       {self.retention:.3f}   "
            f"(share of turn-1 separation the table must keep)",
            *[f"  note: {n}" for n in self.notes],
        ])

    def to_json(self) -> dict:
        return {
            "packet_separation": self.packet_separation,
            "rubric_separation": self.rubric_separation,
            "contestedness": self.contestedness,
            "retention": self.retention,
            "n_seats": self.n_seats,
            "notes": self.notes,
        }


#: Conventions, stated here rather than buried, so they can be argued with and
#: changed in ONE place. None of them is a measurement.
RHO_BASE = 0.60
RHO_W_PACKET = 0.30
RHO_W_CONTEST = 0.20
RHO_MIN, RHO_MAX = 0.40, 0.90


def contestedness(passage_texts: Iterable[str]) -> float:
    """How much the retrieved literature is itself arguing.

    Marker density per 1,000 words, squashed. Crude, and labelled crude: it
    cannot tell a genuine controversy from an author's rhetorical throat-clearing.
    It exists because the alternative -- assuming every question is equally
    contested -- is worse, and demonstrably so: a settled question SHOULD produce
    agreement, and a floor that does not know the difference manufactures
    dissent, which is the failure mode opposite to collapse and just as bad.
    """
    text = " ".join(passage_texts).lower()
    words = max(1, len(text.split()))
    hits = sum(text.count(m) for m in _CONTEST_MARKERS)
    per_k = 1000.0 * hits / words
    return float(1.0 - math.exp(-per_k / 2.0))


def preflight(packets: dict[str, set[str]], rubrics: dict[str, str],
              passage_texts: Iterable[str]) -> Preflight:
    """The divergence prior for one question and one panel.

    `packets` maps seat -> the doc_ids that seat was handed FOR THIS QUESTION.
    That is deliberately question-conditioned: two seats can be globally
    distinct and still be handed near-identical evidence on a particular
    question, and that is precisely when a table collapses.
    """
    names = sorted(packets)
    sets = [packets[n] for n in names]
    pkt = _mean_pairwise(sets, jaccard_distance)

    rub = _mean_pairwise([Counter(tokens(rubrics.get(n, ""))) for n in names],
                         js_divergence)
    con = contestedness(passage_texts)

    rho = RHO_BASE + RHO_W_PACKET * (pkt - 0.5) + RHO_W_CONTEST * (con - 0.5)
    rho = max(RHO_MIN, min(RHO_MAX, rho))

    notes: list[str] = []
    if pkt < 0.35:
        notes.append(
            "these seats were handed largely the SAME evidence for this "
            "question; expect convergence and treat agreement as weak")
    if not any(rubrics.get(n) for n in names):
        notes.append(
            "no seat declared refusals; the objective axis of heterogeneity is "
            "switched off and only the corpus axis is doing any work")
    if con < 0.15:
        notes.append(
            "the retrieved literature shows little sign of active argument; "
            "agreement here may be correct rather than collapsed")
    return Preflight(packet_separation=pkt, rubric_separation=rub,
                     contestedness=con, retention=rho, n_seats=len(names),
                     notes=notes)
