# Example roundtables: idea-flow maps

This directory contains interactive idea-flow maps of three ExpertTwins
roundtables on the immunobiology and immunotherapy of neuroblastoma. Each map
draws the claim-level record of one run. That record includes:

- every claim filed by every seat, grouped into ideas;
- the stances by which seats took up, contested or withdrew those ideas in the
  following turn;
- the moderator question of each turn;
- abstentions with their stated causes;
- citations with their verified quotations;
- peer reviews of the cited evidence.

All three panels are composed of discipline seats. Run labels are those used in
the accompanying manuscript.

| File | Run |
|---|---|
| [`index.html`](index.html) | Overview of the three maps |
| [`NBRT19_flow.html`](NBRT19_flow.html) | Nine discipline seats, open question on the immune landscape |
| [`NBRT14_flow.html`](NBRT14_flow.html) | Four discipline seats, induced collapse |
| [`NBRT16_flow.html`](NBRT16_flow.html) | Four discipline seats, asymmetric base models |

## Opening the maps

GitHub displays HTML files as source text. To view a map, download this
directory (or clone the repository) and open `index.html` or any `*_flow.html`
file in a web browser. Each file is self-contained: the data, the viewer and
the styles are embedded, and the page makes no network requests. NBRT19 is
about 6 MB and may take a few seconds to lay out.

## The runs

| Run | Question | Seats | Base model | Turns | Claims | Ideas | Stances |
|---|---|---|---|---:|---:|---:|---:|
| NBRT19 | The immune landscape of neuroblastoma: what is heterogeneous, why, and what follows | 9 discipline | Opus 5, medium effort | 4 | 431 | 97 | 5,404 |
| NBRT14 | Should dinutuximab be given with GM-CSF in high-risk neuroblastoma? | 4 discipline | Sonnet 5 | 8 | 202 | 49 | 612 |
| NBRT16 | As NBRT14 | 4 discipline | mixed | 3 | 76 | 19 | 144 |

**NBRT19: an open question, nine seats.** The nine seats cover the following
fields:

- T cells;
- cytotoxic T cells;
- regulatory and helper T cells;
- myeloid cells;
- myeloid-derived suppressor cells;
- antigen presentation;
- dendritic cells;
- general immunology;
- neuroblastoma.

The turns proceed as follows:

1. Turn 1 asks the panel to characterise the immune landscape of neuroblastoma
   along several axes of variation. Seats state what is established, contested
   and unknown.
2. Turn 2 asks seats to commit to specifics and to name the observation that
   would refute the emerging convergence.
3. Turn 3 asks seats to answer the adverse verdicts filed against their
   evidence and to defend positions others reject.
4. Turn 4 asks for closing positions in basic science, translational research
   and clinical trials, given by risk group where the answer differs.

Seats recorded 293 abstentions. In the map, contradiction and mixed links
appear from the first transition, and comparatively few ideas are voiced by a
majority of seats.

**NBRT14: an induced collapse.** Four discipline seats took part: immunology,
neuroblastoma, clinical trials and pharmacology.

- Turn 1 was the isolated opening turn.
- Turn 2 used a moderately directive prompt. It urged a common position
  without dictating content, and the panel remained above its floor.
- From turn 3, the moderator prompt dictated six claims verbatim. It
  instructed every seat to file them, to file nothing else, and to agree with
  every other seat. The same prompt was reused through turn 8.

The run is a positive control for the trajectory monitor. The collapse is
manufactured by instruction and does not arise in deliberation. The monitor
classified the dictated turns as collapsing. The guardrail then applied the
five evidence-level rungs of the intervention ladder in order
([docs/06-measurement.md](../../docs/06-measurement.md)). When none restored
separation, it stopped deliberation and reported the collapse.

In the map, the collapse appears from turn 3 as six parallel lanes. Each lane
is voiced by all four seats, and there is no contradiction after turn 2. Ideas
from turn 6 onward are drawn pale because they received no stances.

**NBRT16: asymmetric seat power.** The question, seats, corpus and retrieval
settings were those of NBRT14. Later turns used neutral moderation, which asked
seats to name their main disagreement and not to soften an unconvinced
position. The seats ran on different base models:

| Seat | Base model |
|---|---|
| immunology | Opus 5 |
| clinical trials | Haiku 4.5 |
| neuroblastoma | Gemini 3.5 Flash |
| pharmacology | GPT-5 mini |

The run asks whether a single stronger seat attracts deference without any
instruction to agree. The map shows no collapse in shape: it branches and
carries contradiction. The asymmetry appears in composition instead. The
immunology seat filed 38 of the 76 claims. Its share of each idea can be
followed through the seat-bar fill and the claim counts, by selecting that seat
in the left sidebar. The map does not separate deference from the higher
baseline productivity of the stronger model.

NBRT14 and NBRT16 were recorded with an earlier version of the harness. Their
turn panels therefore show the moderator question and the abstentions, but not
the per-turn monitor metrics.

## Reading a map

Each map is a layered directed graph:

- The seats are the roots.
- Each turn is one layer.
- Each node is an *idea*, a cluster of near-identical claims made in the same
  turn.

| Element | Encoding |
|---|---|
| Node title | A short topic title. The *topic titles* switch replaces it with the first sentence of the idea's most central claim. |
| Node fill and stripe | Theme, a topic that persists across turns. Similar themes have similar hues and are placed side by side. |
| Seat bars | One bar per seat, in fixed order. A filled bar means the seat voiced the idea. The border colour gives that seat's stance on the idea in the next turn: green for support, red for contradiction, purple for mixed, amber for needs other evidence, black for out of scope. |
| Edges | The idea is taken up in the next turn, coloured by the dominant stance. Width increases with the number of stances. |
| Dotted grey edge | A continuation inferred only from semantic similarity, with no declared stance. |
| Dashed black edge | A seat withdrew an earlier claim. |
| Bold or pale | A display heuristic for strength: breadth of authorship and support received, less contradiction and withdrawal. |

The maps support the following interactions:

- **Idea.** Hover over an idea to trace its upstream and downstream lineage.
  Click it to open the evidence panel, which shows:
  - the member claims;
  - the citations, with the quoted passage and its verification status;
  - the peer reviews;
  - the stances received.
- **Turn label.** Click a turn label to see the moderator question and the
  abstentions.
- **Seat.** Click a seat to see its trajectory across turns.
- **Spotlight.** The *spotlight* menu highlights contested, withdrawn, revised,
  merged or split, broadly agreed, or single-seat ideas.
- **Search.** The *search* box matches claims, quotations and references.
- **Layout.** The URL suffixes `#lr` and `#tb` select left-to-right or
  top-down layout.

## Preparation for publication

Only runs composed entirely of discipline seats are included. The following
changes were made to the files; the content of the record was not otherwise
edited.

- **References.**
  - Documents are identified by sequential reference numbers (`ref-001`, …)
    in order of first citation.
  - The reference list embedded in each map gives title, venue, year, DOI and
    PubMed identifier. Author names were removed.
  - A small number of reference numbers appear in stance or abstention text
    without an entry in the list. These are documents that a seat named in
    prose but did not cite in a claim.
- **Author names in text.** Surnames of cited authors were replaced with
  `[author]` wherever they occurred in claims, quotations and stances.
- **Unchanged content.** Claims, quotations, stances, abstentions, peer
  reviews and moderator questions are as produced by the seats and the
  moderator.

## Limitations

- **Embedding clusters.** Ideas and themes are clusters in a sentence-embedding
  space (`all-mpnet-base-v2`), not argument units. A cluster can combine claims
  that share a topic but differ in direction. The evidence panel should be read
  before any conclusion is drawn from a node.
- **Topic titles.** Topic titles and theme labels are summaries written by a
  language model and checked automatically, not by a domain expert. The
  verbatim claims remain available in the tooltip and the evidence panel.
- **Stance resolution.** A stance names a seat, not a claim. Its target is
  therefore resolved heuristically, by similarity to that seat's claims in the
  previous turn. The match score is shown with each stance, and stances below
  0.35 are not drawn.
- **Withdrawals.** Withdrawals are detected from the wording of the claim and
  can miss implicit retractions.
- **Strength.** Strength is a display aid and can make dissent appear weak. The
  *minority voices* spotlight, or turning *strength* off, avoids this.
- **Measurement.** The quantities reported in the manuscript are computed from
  the run records, not from these maps.
- **The visualiser.** The visualiser that produced these maps is not included
  in the repository at present.
