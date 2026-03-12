from typing import Optional, Callable, Any
import re

from app.db.manager import DatabaseManager, normalize_entity_name
from app.ner.base import BaseNERExtractor
from app.services.wiki_explainer import WikipediaExplainer


class KnowledgeBaseService:
    def __init__(
        self,
        db: DatabaseManager,
        extractor: BaseNERExtractor,
        role_nlp: Optional[Callable[[str], Any]] = None,
        explainer: Optional[WikipediaExplainer] = None,
    ):
        self.db = db
        self.extractor = extractor
        self.role_nlp = role_nlp
        self.explainer = explainer or WikipediaExplainer(language="en")
        self.refresh_extractor()

    def refresh_extractor(self) -> None:
        if hasattr(self.extractor, "load_patterns_from_db"):
            self.extractor.load_patterns_from_db(self.db)

    def _clean_text(self, text: str) -> str:
        text = text.strip()
        text = re.sub(r"\s+", " ", text)
        return text

    def _is_valid_entity_text(self, value: str) -> bool:
        value = value.strip()
        if not value:
            return False
        if len(value) == 1 and not value.isalpha():
            return False
        return True

    def _guess_role(
        self,
        entity_text: str,
        full_text: str,
        span_start: Optional[int] = None,
        span_end: Optional[int] = None,
        doc=None,
    ) -> str:
        """
        brief: Heuristically determine entity role using dependency parse when available.
        """
        text_lower = full_text.lower()
        ent_lower = entity_text.lower()

        # Try to use dependency parse if spaCy doc is provided
        if doc is not None and span_start is not None and span_end is not None:
            subject_deps = {"nsubj", "nsubjpass", "csubj", "csubjpass"}
            object_deps = {"dobj", "obj", "iobj", "pobj", "dative", "attr", "oprd"}

            found_subject = False
            found_object = False

            for token in doc:
                tok_start = token.idx
                tok_end = token.idx + len(token)

                # Overlap between token span and entity span
                if tok_start < span_end and tok_end > span_start:
                    if token.dep_ in subject_deps:
                        found_subject = True
                    if token.dep_ in object_deps:
                        found_object = True

            if found_subject:
                return "subject"
            if found_object:
                return "object"

        # Fallback heuristic: entity at the beginning of the text
        if text_lower.startswith(ent_lower + " "):
            return "subject"
        return "object"

    def _wiki_matches_category(self, wiki_summary: str, category_name: str) -> bool:
        """
        brief: Simple keyword-based classifier that decides whether a wikipedia
               summary corresponds to a given fine-grained NER category.
        """
        if not wiki_summary:
            return False

        text = wiki_summary.lower()
        category = category_name.lower()

        keywords_by_category = {
            "animal": [
                "animal",
                "species",
                "mammal",
                "bird",
                "reptile",
                "fish",
                "insect",
                "amphibian",
                "breed",
                "domesticated",
                "wildlife",
                "vertebrate",
            ],
            "job_type": [
                "profession",
                "occupation",
                "job",
                "career",
                "title",
                "role",
                "position",
                "employed as",
                "works as",
                "is a",
            ],
            "name_of_art": [
                "novel",
                "poem",
                "poetry",
                "painting",
                "sculpture",
                "film",
                "movie",
                "opera",
                "symphony",
                "song",
                "album",
                "play",
                "ballet",
                "artwork",
                "work of art",
                "short story",
                "painting by",
                "film directed",
                "song by",
            ],
        }

        keywords = keywords_by_category.get(category, [])
        if not keywords:
            return category in text

        return any(kw in text for kw in keywords)

    def build_category_from_local_descriptions(
        self,
        category_name: str,
        source_categories: Optional[list[str]] = None,
        limit: Optional[int] = None,
        fetch_missing_descriptions: bool = True,
    ) -> dict:
        """
        brief: Create a fine-grained NER category and
        automatically populate it using only local information
        (entity names/descriptions) with simple keyword rules.

        return[out] Dict with basic statistics of the operation.
        """
        # Ensure category exists
        created_category_id = self.db.add_category(
            category_name,
            description=f"Auto-built fine-grained category '{category_name}' from local rules",
        )

        # Collect candidate entities
        if source_categories:
            candidates: list[dict] = []
            for cat_name in source_categories:
                candidates.extend(self.db.list_entities(category_name=cat_name))
        else:
            candidates = self.db.list_entities()

        processed = 0
        reassigned = 0
        skipped_no_text = 0
        skipped_mismatch = 0

        for ent in candidates:
            if limit is not None and processed >= limit:
                break

            processed += 1
            name = ent["name"]
            text = ent.get("description") or ""

            # fetch description once if missing
            if not text and fetch_missing_descriptions and self.explainer is not None:
                wiki_info = self.explainer.explain(name)
                if wiki_info.get("found") and wiki_info.get("summary"):
                    text = wiki_info["summary"]
                    # cache it in DB for future use
                    self.db.update_entity_description(
                        entity_id=ent["id"],
                        description=wiki_info["summary"],
                        description_source="wikipedia",
                        wiki_url=wiki_info.get("url") or "",
                    )

            if not text:
                skipped_no_text += 1
                continue

            if not self._wiki_matches_category(text, category_name):
                skipped_mismatch += 1
                continue

            if ent["category"] == category_name:
                # already assigned, nothing to do
                continue

            self.db.reassign_entity(entity_name=name, new_category_name=category_name)
            reassigned += 1

        self.refresh_extractor()

        return {
            "category": category_name,
            "category_id": created_category_id,
            "processed_entities": processed,
            "reassigned_entities": reassigned,
            "skipped_no_text": skipped_no_text,
            "skipped_mismatch": skipped_mismatch,
        }

    def add_category(self, name: str, description: str = "") -> int:
        category_id = self.db.add_category(name, description)
        self.refresh_extractor()
        return category_id

    def add_entity_manual(self, name: str, category_name: str, description: str = "") -> int:
        entity_id = self.db.add_entity(name=name, category_name=category_name, description=description)
        self.refresh_extractor()
        return entity_id

    def delete_entity(self, name: str, category_name: Optional[str] = None) -> bool:
        deleted = self.db.delete_entity(name=name, category_name=category_name)
        if deleted:
            self.refresh_extractor()
        return deleted

    def ingest_text(
        self,
        text: str,
        source: str = "",
        auto_add_unknown: bool = True,
    ) -> dict:
        cleaned_text = self._clean_text(text)
        if not cleaned_text:
            return {
                "text_id": None,
                "entities": [],
                "new_entities": [],
                "message": "Empty text"
            }

        text_id = self.db.add_text(cleaned_text, source=source)
        extracted = self.extractor.extract(cleaned_text)

        # build spaCy doc once if we have a parser available.
        doc = None

        nlp = self.role_nlp
        if nlp is None:
            nlp = getattr(self.extractor, "nlp", None)

        if nlp is not None:
            try:
                doc = nlp(cleaned_text)
            except Exception:
                doc = None

        saved_entities = []
        new_entities = []
        seen_pairs = set()

        for ent in extracted:
            entity_name = ent.text.strip()
            category_name = ent.label.strip()

            if not self._is_valid_entity_text(entity_name):
                continue

            pair_key = (normalize_entity_name(entity_name), category_name)
            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)

            category = self.db.get_category(category_name)
            if category is None:
                if auto_add_unknown:
                    self.db.add_category(
                        category_name,
                        description=f"Auto-created from extractor label {category_name}"
                    )
                else:
                    continue

            existing = self.db.get_entity(entity_name, category_name)

            if existing is None:
                entity_id = self.db.add_entity(
                    name=entity_name,
                    category_name=category_name,
                    description=f"Auto-added from parsed text: [{ent.start}, {ent.end}]"
                )
                entity_data = self.db.get_entity_by_id(entity_id)
                self.refresh_extractor()
                new_entities.append({
                    "id": entity_data["id"],
                    "name": entity_data["name"],
                    "category": entity_data["category"],
                })
            else:
                entity_id = existing["id"]
                entity_data = existing

            role = self._guess_role(
                entity_text=entity_name,
                full_text=cleaned_text,
                span_start=ent.start,
                span_end=ent.end,
                doc=doc,
            )

            self.db.link_entity_to_text(
                entity_id=entity_id,
                text_id=text_id,
                role=role
            )

            saved_entities.append({
                "id": entity_data["id"],
                "name": entity_data["name"],
                "category": entity_data["category"],
                "role": role,
                "start": ent.start,
                "end": ent.end,
            })

        return {
            "text_id": text_id,
            "entities": saved_entities,
            "new_entities": new_entities,
            "message": f"Stored text and linked {len(saved_entities)} entities"
        }

    def get_texts_for_entity(self, entity_name: str, role: Optional[str] = None) -> list[dict]:
        return self.db.get_texts_for_entity(entity_name, role=role)

    def get_entity_overview(self, entity_name: str) -> dict:
        matches = self.db.search_entity(entity_name)
        if not matches:
            return {
                "found": False,
                "entity": None,
                "texts": [],
                "stats": {},
                "message": f"Entity '{entity_name}' not found"
            }

        entity = matches[0]
        texts = self.db.get_texts_for_entity(entity["name"])

        role_counts = {}
        for item in texts:
            role = item["role"]
            role_counts[role] = role_counts.get(role, 0) + 1

        return {
            "found": True,
            "entity": entity,
            "texts": texts,
            "stats": {
                "text_count": len(texts),
                "role_counts": role_counts,
            },
            "message": f"Found {len(texts)} related texts"
        }

    def stats(self) -> dict:
        return self.db.stats()

    def get_category_overview(self, category_name: str) -> dict:
        category = self.db.get_category(category_name)
        if not category:
            return {"found": False, "message": f"Category '{category_name}' not found"}

        entities = self.db.list_entities(category_name=category_name)

        enriched = []
        total_links = 0
        for ent in entities:
            texts = self.db.get_texts_for_entity(ent["name"])
            total_links += len(texts)
            enriched.append({
                "id": ent["id"],
                "name": ent["name"],
                "description": ent.get("description", ""),
                "text_count": len(texts),
            })

        enriched.sort(key=lambda x: x["text_count"], reverse=True)

        return {
            "found": True,
            "category": category,
            "entity_count": len(entities),
            "text_links_count": total_links,
            "entities": enriched,
        }
    
    def get_explanation(self, entity_name: str) -> dict:
        matches = self.db.search_entity(entity_name)
        if not matches:
            return {
                "found": False,
                "entity": entity_name,
                "message": "Entity not found",
            }

        entity = matches[0]

        if entity.get("description"):
            return {
                "found": True,
                "entity": entity["name"],
                "category": entity["category"],
                "explanation": entity["description"],
                "source": entity.get("description_source") or "database",
                "wiki_url": entity.get("wiki_url"),
                "cached": True,
            }

        wiki_info = self.explainer.explain(entity["name"])

        if wiki_info["found"] and wiki_info["summary"]:
            self.db.update_entity_description(
                entity_id=entity["id"],
                description=wiki_info["summary"],
                description_source="wikipedia",
                wiki_url=wiki_info["url"] or "",
            )

            return {
                "found": True,
                "entity": entity["name"],
                "category": entity["category"],
                "explanation": wiki_info["summary"],
                "source": "wikipedia",
                "wiki_url": wiki_info["url"],
                "cached": False,
            }

        return {
            "found": False,
            "entity": entity["name"],
            "category": entity["category"],
            "message": "Explanation not found in Wikipedia",
        }

    def build_category_from_kb(
        self,
        category_name: str,
        source_categories: Optional[list[str]] = None,
        limit: Optional[int] = None,
    ) -> dict:
        # Ensure category exists
        created_category_id = self.db.add_category(
            category_name,
            description=f"Auto-built fine-grained category '{category_name}' from KB",
        )

        # Collect candidate entities
        if source_categories:
            candidates: list[dict] = []
            for cat_name in source_categories:
                candidates.extend(self.db.list_entities(category_name=cat_name))
        else:
            candidates = self.db.list_entities()

        processed = 0
        reassigned = 0
        skipped_no_wiki = 0
        skipped_mismatch = 0

        for ent in candidates:
            if limit is not None and processed >= limit:
                break

            processed += 1
            name = ent["name"]

            wiki_info = self.explainer.explain(name)
            if not wiki_info.get("found") or not wiki_info.get("summary"):
                skipped_no_wiki += 1
                continue

            summary = wiki_info["summary"]
            if not self._wiki_matches_category(summary, category_name):
                skipped_mismatch += 1
                continue

            if ent["category"] == category_name:
                # already assigned, nothing to do
                continue

            # Reassign entity to the new category
            self.db.reassign_entity(entity_name=name, new_category_name=category_name)
            reassigned += 1

        # Refresh extractor patterns once at the end
        self.refresh_extractor()

        return {
            "category": category_name,
            "category_id": created_category_id,
            "processed_entities": processed,
            "reassigned_entities": reassigned,
            "skipped_no_wiki": skipped_no_wiki,
            "skipped_mismatch": skipped_mismatch,
        }
