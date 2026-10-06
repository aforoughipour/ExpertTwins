"""`Library.verify()` against a corpus written by an independent loader.

The toy generator in examples/toy writes the library format with the standard
library alone, so it is a second implementation of the store's contract. If the
two disagree -- on the passage-id format, the text hash, or the counts -- the
toy corpus fails its own audit, and so would any loader written from it.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from expertwins.corpus.store import Library

ROOT = Path(__file__).resolve().parents[1]


def _toy_module():
    path = ROOT / "examples" / "toy" / "make_toy_corpus.py"
    spec = importlib.util.spec_from_file_location("make_toy_corpus", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture()
def toy_library(tmp_path: Path) -> Path:
    mod = _toy_module()
    root = tmp_path / "library"
    docs = mod.make_docs()
    mod.write_library(root, docs)
    mod.build_index(root, docs)
    return root


def _rewrite_first_passage(text_jsonl: Path, **changes: str) -> None:
    lines = text_jsonl.read_text(encoding="utf-8").splitlines()
    row = json.loads(lines[0])
    row.update(changes)
    lines[0] = json.dumps(row, ensure_ascii=False)
    text_jsonl.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_toy_corpus_passes_the_store_audit(toy_library: Path):
    report = Library(toy_library).verify()
    assert report.checked == 18
    assert report.errors == 0, report.render()
    assert report.warnings == 0, report.render()


def test_toy_passage_ids_use_the_store_format(toy_library: Path):
    lib = Library(toy_library)
    for doc_id in lib.current_rows():
        for p in lib.load_passages(doc_id):
            assert p.passage_id == f"{doc_id}#p{p.order:04d}"


def test_passage_naming_another_document_is_an_error(toy_library: Path):
    files = sorted((toy_library / "docs").rglob("text.jsonl"))
    victim, other = files[0], files[1]
    _rewrite_first_passage(victim, passage_id=f"{other.parent.name}#p0000")
    report = Library(toy_library).verify()
    assert [m.split(" ", 1)[0] for m in report.passage_id_mismatch] == [victim.parent.name]
    assert report.errors == 1
    # The hash covers text only, so it does not see this defect.
    assert report.text_hash_mismatch == []


def test_edited_passage_text_is_an_error(toy_library: Path):
    victim = sorted((toy_library / "docs").rglob("text.jsonl"))[0]
    row = json.loads(victim.read_text(encoding="utf-8").splitlines()[0])
    _rewrite_first_passage(victim, text=row["text"] + " edited")
    report = Library(toy_library).verify()
    assert report.text_hash_mismatch == [victim.parent.name]
    assert report.errors >= 1
