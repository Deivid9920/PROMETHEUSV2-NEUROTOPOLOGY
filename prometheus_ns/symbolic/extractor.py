"""Unsupervised triplet extraction with spaCy dependency patterns.

Three relation patterns are extracted per sentence:

- copula: "X is a Y" -> ``(X, is_a, Y)`` (subject + be-root + attribute);
- active verb: nsubj + VERB + dobj -> ``(subj, verb, obj)``;
- passive: the ``nsubjpass`` dependency marks the patient and the
  ``agent`` prepositional phrase carries the true subject, so
  "The mouse was chased by the cat" yields (cat, chase, mouse) and not
  the inverted (mouse, chase, cat);
- apposition: "X, a Y," -> ``(X, is_a, Y)``.

Entities are normalized (lowercase, determiner stripping, lemma of the
head word) and empty, stopword-only or pronominal entities are filtered.
Each triplet carries a confidence and the SHA-256 of its source doc.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path

import spacy
from spacy.language import Language
from spacy.tokens import Doc, Token

from prometheus_ns import cfg_get, load_config, repo_path

_STOPWORD_ENTITIES = frozenset({"this", "that", "these", "those", "there", "here", "it", "he", "she", "they", "we", "you", "i", "one", "someone", "something", "anything", "nothing", "everyone"})


@dataclass(frozen=True)
class Triplet:
    """One extracted (subject, relation, object) fact with provenance."""

    sub: str
    rel: str
    obj: str
    conf: float
    source_doc_hash: str


def load_nlp(cfg: dict, n_process: int = 1) -> Language:
    """Load the configured spaCy model with the pipes this extractor needs."""
    model_name = str(cfg_get(cfg, "symbolic.spacy_model", "en_core_web_sm"))
    nlp = spacy.load(model_name)
    if n_process > 1:
        nlp.max_length = max(nlp.max_length, 2_000_000)
    return nlp


def normalize_entity(span_text: str, head: Token | None = None) -> str:
    """Normalize an entity surface form for graph matching.

    Lowercases, strips leading determiners and possessives, and prefers
    the lemma of the syntactic head when available. This is the
    polysemy guard: normalized forms are what the consistency graph
    compares.
    """
    words = span_text.strip().lower().split()
    while words and words[0] in {"the", "a", "an", "his", "her", "its", "their"}:
        words = words[1:]
    lemma = head.lemma_.lower().strip() if head is not None and head.lemma_ else ""
    if lemma and lemma not in _STOPWORD_ENTITIES:
        return lemma if len(lemma) > 2 else " ".join(words)
    return " ".join(words)


def _is_valid_entity(text: str) -> bool:
    if len(text) < 2 or len(text) > 60:
        return False
    if text in _STOPWORD_ENTITIES:
        return False
    if not any(ch.isalnum() for ch in text):
        return False
    return True


def _entity_text(token: Token) -> tuple[str, Token]:
    """Return the normalized text and head of the noun phrase around ``token``."""
    span = token.subtree if token.dep_ in ("nsubj", "nsubjpass", "dobj", "attr", "appos", "pobj") else token
    text = " ".join(t.text for t in span if not t.is_punct and not t.is_space)
    if not text:
        text = token.text
    return normalize_entity(text, head=token), token


def extract_triplets_from_doc(
    doc: Doc,
    min_confidence: float,
    source_doc_hash: str = "",
) -> list[Triplet]:
    """Extract triplets from one parsed spaCy document."""
    triplets: list[Triplet] = []

    for sent in doc.sents:
        root = sent.root
        for verb in [root, *[t for t in sent if t.dep_ == "conj" and t.pos_ == "VERB"]]:
            if verb.pos_ != "VERB":
                continue
            auxpass = any(child.dep_ == "auxpass" for child in verb.children)
            subjects = [c for c in verb.children if c.dep_ == "nsubj"]
            pass_subjects = [c for c in verb.children if c.dep_ == "nsubjpass"]
            dobj = [c for c in verb.children if c.dep_ in ("dobj", "obj")]

            if auxpass and pass_subjects:
                # Passive voice: the agent sits in the "by"-phrase.
                agent = None
                for child in verb.children:
                    if child.dep_ == "agent":
                        pobjs = [c for c in child.children if c.dep_ == "pobj"]
                        if pobjs:
                            agent = pobjs[0]
                patient = pass_subjects[0]
                if agent is not None:
                    sub, _ = _entity_text(agent)
                    obj, _ = _entity_text(patient)
                    if _is_valid_entity(sub) and _is_valid_entity(obj):
                        triplets.append(Triplet(sub, verb.lemma_.lower(), obj, 0.65, source_doc_hash))
            elif subjects and dobj:
                sub, _ = _entity_text(subjects[0])
                obj, _ = _entity_text(dobj[0])
                if _is_valid_entity(sub) and _is_valid_entity(obj):
                    triplets.append(Triplet(sub, verb.lemma_.lower(), obj, 0.7, source_doc_hash))

        # Copula: "X is a Y" with attribute, or adjectival property.
        be_children_nsubj = [t for t in sent if t.lemma_ == "be" and t.pos_ in ("AUX", "VERB")]
        for be in be_children_nsubj:
            nsubjs = [c for c in be.children if c.dep_ == "nsubj"]
            attrs = [c for c in be.children if c.dep_ == "attr"]
            if nsubjs and attrs:
                sub, _ = _entity_text(nsubjs[0])
                obj, _ = _entity_text(attrs[0])
                if _is_valid_entity(sub) and _is_valid_entity(obj):
                    triplets.append(Triplet(sub, "is_a", obj, 0.9, source_doc_hash))

        # Apposition: "X, a Y, ..." -> X is_a Y.
        for token in sent:
            if token.dep_ == "appos" and token.head.dep_ in ("nsubj", "nsubjpass", "dobj", "pobj", "attr"):
                sub, _ = _entity_text(token.head)
                obj, _ = _entity_text(token)
                if _is_valid_entity(sub) and _is_valid_entity(obj):
                    triplets.append(Triplet(sub, "is_a", obj, 0.6, source_doc_hash))

    return [t for t in triplets if t.conf >= min_confidence]


def extract_corpus(cfg: dict, docs_dir: Path | None = None, n_process: int = 1) -> dict:
    """Extract triplets over the cleaned corpus (excluding the holdout).

    Writes ``triplets/triplets.jsonl`` and per-run stats to
    ``logs/symbolic_stats.json``. spaCy parsing is the slowest stage of
    a round on CPU; ``n_process`` parallelizes it (``nlp.pipe``).
    """
    clean_dir = repo_path(cfg, "data_clean")
    docs_dir = docs_dir or clean_dir
    holdout = clean_dir / "holdout"
    excluded = {p.resolve() for p in holdout.glob("**/*.txt")}
    files = sorted(p for p in docs_dir.glob("**/*.txt") if p.resolve() not in excluded)

    min_confidence = float(cfg_get(cfg, "symbolic.min_confidence", 0.6))
    nlp = load_nlp(cfg)

    out_dir = repo_path(cfg, "triplets")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "triplets.jsonl"

    kept = 0
    low_confidence = 0
    n_docs = 0
    with out_path.open("w", encoding="utf-8") as out:
        for path in files:
            text = path.read_text(encoding="utf-8")
            if not text.strip():
                continue
            n_docs += 1
            dhash = sha256(text.encode("utf-8")).hexdigest()
            doc = nlp(text)
            for triplet in extract_triplets_from_doc(doc, min_confidence, dhash):
                out.write(json.dumps(asdict(triplet), ensure_ascii=False) + "\n")
                kept += 1

    stats = {"docs": n_docs, "triplets": kept, "min_confidence": min_confidence, "files": len(files)}
    stats_path = repo_path(cfg, "logs", "symbolic_stats.json")
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return stats


def main() -> None:
    """CLI entry point for corpus-wide triplet extraction."""
    import argparse

    parser = argparse.ArgumentParser(description="Prometheus-NS symbolic triplet extraction")
    parser.add_argument("--config", required=True, help="path to config.yaml")
    parser.add_argument("--n-process", type=int, default=1, help="spaCy n_process for parallel parsing")
    args = parser.parse_args()
    cfg = load_config(args.config)
    print(json.dumps(extract_corpus(cfg, n_process=args.n_process), indent=2))


if __name__ == "__main__":
    main()
