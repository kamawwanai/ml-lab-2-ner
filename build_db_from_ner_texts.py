from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

from app.db import DatabaseManager
from app.ner import HFTokenClassificationExtractor
from app.services.kb_service import KnowledgeBaseService


def _normalize_model_label(raw: str) -> Optional[str]:
    """
    Convert model label to DB category name.
    - strip B-/I- prefix if present
    - lowercase
    """
    if not raw:
        return None
    label = str(raw).strip()
    if not label:
        return None

    # common: "B-per", "I-geo"
    if "-" in label:
        prefix, rest = label.split("-", 1)
        if prefix.upper() in {"B", "I"} and rest.strip():
            label = rest

    label = label.strip().lower()
    return label or None


def _iter_texts(txt_path: Path) -> Iterable[tuple[int, str]]:
    """
    Yield (line_no, text) for non-empty lines (1-indexed).
    """
    with txt_path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            text = line.strip()
            if not text:
                continue
            yield i, text


@dataclass(frozen=True)
class _PredEntity:
    name: str
    category: str
    start: int
    end: int


def _unique_entities(spans) -> list[_PredEntity]:
    uniq: list[_PredEntity] = []
    seen: set[tuple[str, str, int, int]] = set()

    for s in spans:
        name = str(getattr(s, "text", "")).strip()
        cat = _normalize_model_label(getattr(s, "label", ""))
        start = getattr(s, "start", None)
        end = getattr(s, "end", None)
        if not name or not cat:
            continue
        if start is None or end is None:
            continue
        try:
            start_i = int(start)
            end_i = int(end)
        except Exception:
            continue
        if start_i < 0 or end_i <= start_i:
            continue

        key = (name.lower(), cat, start_i, end_i)
        if key in seen:
            continue
        seen.add(key)
        uniq.append(_PredEntity(name=name, category=cat, start=start_i, end=end_i))

    return uniq


def populate_db_from_texts_with_model(
    db_path: Path,
    texts_path: Path,
    model_dir: str = "models/my_ner_model",
    device: int = -1,
    aggregation_strategy: str = "simple",
    source_name: str = "ner_texts_200.txt",
) -> None:
    texts_path = texts_path.resolve()
    if not texts_path.exists():
        raise FileNotFoundError(f"Texts file not found: {texts_path}")

    if db_path.exists():
        db_path.unlink()

    try:
        extractor = HFTokenClassificationExtractor(
            model_dir=model_dir,
            device=device,
            aggregation_strategy=aggregation_strategy,
        )
    except ModuleNotFoundError as e:
        if getattr(e, "name", "") in {"transformers", "torch"}:
            raise ModuleNotFoundError(
                "Не найдены зависимости для HuggingFace модели. "
                "Установите зависимости из requirements.txt (нужны как минимум `transformers` и `torch`)."
            ) from e
        raise

    db = DatabaseManager(str(db_path))
    kb = KnowledgeBaseService(db=db, extractor=extractor)

    text_count = 0
    ent_total = 0

    for line_no, text in _iter_texts(texts_path):
        text_id = db.add_text(text, source=f"{source_name}:L{line_no}")
        spans = extractor.extract(text)
        entities = _unique_entities(spans)

        # Try to use spaCy dependency parse for roles (subject/object).
        doc = None
        try:
            import spacy  # type: ignore

            doc = spacy.load("en_core_web_sm")(text)
        except Exception:
            doc = None

        for ent in entities:
            db.add_category(ent.category, description="")
            db.add_entity(name=ent.name, category_name=ent.category)
            row = db.get_entity(ent.name, ent.category)
            if row is None:
                raise RuntimeError(
                    f"Entity not found right after insert: name={ent.name!r}, category={ent.category!r}, line={line_no}"
                )
            role = kb._guess_role(
                entity_text=ent.name,
                full_text=text,
                span_start=ent.start,
                span_end=ent.end,
                doc=doc,
            )
            db.link_entity_to_text(row["id"], text_id, role=role)

        text_count += 1
        ent_total += len(entities)

        if text_count % 50 == 0:
            print(f"Processed {text_count} texts, {ent_total} entities linked...")

    print(f"Done. Texts: {text_count}, entities linked: {ent_total}.")
    print("DB stats:", db.stats())
    db.close()


def main() -> None:
    p = argparse.ArgumentParser(
        description="Build ner_kb.db from ner_texts_200.txt using a fine-tuned HF NER model."
    )
    p.add_argument(
        "--texts",
        type=str,
        default="ner_texts_200.txt",
        help="Path to input texts file (one text per line).",
    )
    p.add_argument(
        "--db",
        type=str,
        default="data/ner_kb.db",
        help="Path to output sqlite DB.",
    )
    p.add_argument(
        "--model-dir",
        type=str,
        default="models/my_ner_model",
        help="Path to HF token-classification model directory.",
    )
    p.add_argument(
        "--device",
        type=int,
        default=-1,
        help="Device for transformers pipeline (-1=CPU, 0=CUDA:0, ...).",
    )
    p.add_argument(
        "--aggregation-strategy",
        type=str,
        default="simple",
        help="Transformers pipeline aggregation_strategy (e.g. simple/first/max/average).",
    )
    args = p.parse_args()

    project_root = Path(__file__).resolve().parent
    populate_db_from_texts_with_model(
        db_path=(project_root / args.db),
        texts_path=(project_root / args.texts),
        model_dir=args.model_dir,
        device=args.device,
        aggregation_strategy=args.aggregation_strategy,
        source_name=Path(args.texts).name,
    )


if __name__ == "__main__":
    main()

