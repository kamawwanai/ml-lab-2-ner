import csv
from collections import defaultdict
from pathlib import Path

from app.db import DatabaseManager
import sqlite3
from app.services.kb_service import KnowledgeBaseService


ID2TAG = {
    0: "O",
    1: "B-per",
    2: "I-per",
    3: "B-gpe",
    4: "I-gpe",
    5: "B-eve",
    6: "I-eve",
    7: "B-geo",
    8: "I-geo",
    9: "B-nat",
    10: "I-nat",
    11: "B-art",
    12: "I-art",
    13: "B-tim",
    14: "I-tim",
    15: "B-org",
    16: "I-org",
}


TYPE2CATEGORY = {
    # categories in the DB will exactly match model tags
    # (surface types without B-/I- prefix)
    "per": "per",
    "gpe": "gpe",
    "eve": "eve",
    "geo": "geo",
    "nat": "nat",
    "art": "art",
    "tim": "tim",
    "org": "org",
}


def decode_tag(tag_id: int):
    """
    Convert numeric tag id to (prefix, entity_type) or None for 'O'.
    Example: 1 -> ('B', 'per')
    """
    tag = ID2TAG.get(tag_id, "O")
    if tag == "O":
        return None
    prefix, etype = tag.split("-", 1)
    return prefix, etype


def iter_sentences(csv_path: Path):
    """
    Yield (sentence_id, tokens, tag_ids) from train.csv
    """
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)

        current_sid = None
        tokens = []
        tags = []

        for row in reader:
            sid_raw = row["Sentence_id"]
            # Sentence_id заполнен только для первого токена предложения
            if sid_raw:
                sid = sid_raw
            else:
                sid = current_sid

            if current_sid is None:
                current_sid = sid

            # новая последовательность предложения
            if sid != current_sid:
                if tokens:
                    yield current_sid, tokens, tags
                current_sid = sid
                tokens = []
                tags = []

            word = row["Word"]
            tag_id = int(row["Tag"])
            tokens.append(word)
            tags.append(tag_id)

        # последняя накопленная последовательность
        if tokens:
            yield current_sid, tokens, tags


def extract_entities(tokens, tag_ids):
    """
    From tokens and tag_ids (IOB2 numeric) produce list of
    (text, category_name, start_token_idx, end_token_idx_exclusive).
    """
    entities = []
    start = None
    cur_type = None

    for i, tag_id in enumerate(tag_ids + [0]):  # sentinel O в конце
        decoded = decode_tag(tag_id)

        if start is None:
            # не внутри сущности
            if decoded is None:
                continue
            # начинаем новую сущность
            prefix, etype = decoded
            start = i
            cur_type = etype
        else:
            # мы внутри сущности
            if decoded is None:
                # сущность закончилась
                end = i
                ent_tokens = tokens[start:end]
                text = " ".join(ent_tokens).strip()
                cat = TYPE2CATEGORY.get(cur_type)
                if text and cat:
                    entities.append((text, cat, start, end))
                start = None
                cur_type = None
            else:
                prefix, etype = decoded
                if prefix == "I" and etype == cur_type:
                    # продолжаем ту же сущность
                    continue
                else:
                    # закрываем предыдущую и начинаем новую
                    end = i
                    ent_tokens = tokens[start:end]
                    text = " ".join(ent_tokens).strip()
                    cat = TYPE2CATEGORY.get(cur_type)
                    if text and cat:
                        entities.append((text, cat, start, end))

                    start = i
                    cur_type = etype

    # убираем дубликаты внутри предложения
    unique = []
    seen = set()
    for text, cat, st, en in entities:
        key = (text.lower(), cat, st, en)
        if key in seen:
            continue
        seen.add(key)
        unique.append((text, cat, st, en))

    return unique


def _build_sentence_and_token_char_offsets(tokens: list[str]) -> tuple[str, list[tuple[int, int]]]:
    """
    Returns sentence and per-token (start,end) char offsets in that sentence.
    Sentence is built as " ".join(tokens), matching existing logic.
    """
    parts = []
    offsets: list[tuple[int, int]] = []
    cursor = 0
    for i, tok in enumerate(tokens):
        if i > 0:
            parts.append(" ")
            cursor += 1
        start = cursor
        parts.append(tok)
        cursor += len(tok)
        end = cursor
        offsets.append((start, end))
    return "".join(parts).strip(), offsets


def populate_db_from_csv(
    db_path: Path = Path("data/ner_kb.db"),
    csv_path: Path = Path("train.csv"),
):
    csv_path = csv_path.resolve()
    db_path = db_path

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    # Всегда создаём БД с нуля для чистого импорта
    if db_path.exists():
        db_path.unlink()

    db = DatabaseManager(str(db_path))
    kb = KnowledgeBaseService(db=db, extractor=None)  # type: ignore[arg-type]

    # Try to load spaCy parser for dependency roles.
    role_nlp = None
    try:
        import spacy  # type: ignore

        role_nlp = spacy.load("en_core_web_sm")
    except Exception:
        role_nlp = None

    # заранее гарантируем наличие категорий (без спец-описаний)
    for cat in sorted(set(TYPE2CATEGORY.values())):
        db.add_category(cat, description="")

    sent_count = 0
    ent_total = 0

    for sid, tokens, tag_ids in iter_sentences(csv_path):
        sentence, token_offsets = _build_sentence_and_token_char_offsets(tokens)
        if not sentence:
            continue

        text_id = db.add_text(sentence, source="train.csv")
        entities = extract_entities(tokens, tag_ids)

        doc = None
        if role_nlp is not None:
            try:
                doc = role_nlp(sentence)
            except Exception:
                doc = None

        for name, category, st_tok, en_tok in entities:
            # Гарантируем, что сущность существует и берём её id «по факту» из БД
            db.add_entity(
                name=name,
                category_name=category,
            )
            ent = db.get_entity(name, category)
            if ent is None:
                # Если такое случится — это уже логическая ошибка, лучше явно сообщить
                raise RuntimeError(
                    f"Entity not found right after insert: name={name!r}, category={category!r}, sid={sid}"
                )

            # Связь создаётся через безопасный метод, который сам проверяет FK
            span_start = token_offsets[st_tok][0] if 0 <= st_tok < len(token_offsets) else None
            span_end = token_offsets[en_tok - 1][1] if 0 < en_tok <= len(token_offsets) else None
            role = kb._guess_role(
                entity_text=name,
                full_text=sentence,
                span_start=span_start,
                span_end=span_end,
                doc=doc,
            )
            db.link_entity_to_text(ent["id"], text_id, role=role)

        sent_count += 1
        ent_total += len(entities)

        if sent_count % 1000 == 0:
            print(f"Processed {sent_count} sentences, {ent_total} entities total...")

    print(f"Done. Sentences: {sent_count}, entities linked: {ent_total}.")
    print("DB stats:", db.stats())
    db.close()


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent
    populate_db_from_csv(
        db_path=project_root / "data" / "ner_kb.db",
        csv_path=project_root / "train.csv",
    )

