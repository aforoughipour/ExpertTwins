"""The corpus layer: literature acquisition, the document store and its index.

The store is primary; the index and everything downstream of it are derived and
can be rebuilt from it. Two integrity properties of this layer are relied upon
by the rest of the package:

  * `store.classify_fulltext` -- the store measures the stored bytes and decides
    whether a document is full text. No caller may assert it.
  * `store.title_appears_in_text` -- screens out a stored document that is a
    different paper citing the intended one, with every hash matching.

Acquisition (`acquire/`, `net`, `extract`) is the only part of the package that
uses the network; the read path imports only `store`, `index` and their
dependencies.
"""
