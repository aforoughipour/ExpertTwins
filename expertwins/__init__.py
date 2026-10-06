"""ExpertTwins -- evidence-grounded scientific panel simulation.

ExpertTwins supports two kinds of seats. Person seats are modelled on an
individual scientist's own publications and citation neighbourhood. Discipline
seats are modelled on a field's literature. Each seat receives a private
evidence packet, and the panel records what each seat was actually allowed to
read.

The core doctrine:

    derive, never assert          the store measures the bytes and decides
    character-exact verification  a quote is in the paper or it is not
    permitted-set membership      a citation outside what a seat was handed is
                                  a provenance failure, checked by set
                                  membership rather than by judgement
    typed abstention              absence and failure are different types

The additions, and what each is for:

    identity.py    a person is a corpus, a lineage, a set of standing refusals
                   and a repertoire of moves. Citations are
                   TIERED into `own` (they wrote it) and `read` (they were
                   handed it), which makes provenance mechanically checkable.
    diversity.py   several axes of heterogeneity, measured per turn against
                   permutation nulls.
    guardrail.py   an adaptive floor under that heterogeneity, and a graduated
                   ladder of interventions that act on EVIDENCE rather than on
                   tone -- because instructing an agent to disagree produces
                   disagreement-shaped text, not disagreement.
    tasks.py       the panel does more than argue: reviews, critiques, talks,
                   and analyses, each with its own deliverable contract.
"""

__version__ = "1.0.0"
