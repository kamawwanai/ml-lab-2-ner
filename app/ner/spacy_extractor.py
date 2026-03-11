from typing import List, Optional
import spacy

from app.ner.base import BaseNERExtractor, EntitySpan


class SpacyExtractor(BaseNERExtractor):
    def __init__(self, model_name: str = "en_core_web_sm"):
        try:
            self.nlp = spacy.load(model_name)
        except Exception:
            self.nlp = spacy.blank("en")

        self._build_empty_ruler()

    def _build_empty_ruler(self) -> None:
        if "entity_ruler" in self.nlp.pipe_names:
            self.nlp.remove_pipe("entity_ruler")

        config = {
            "overwrite_ents": True,
            "phrase_matcher_attr": "LOWER",
        }

        if "ner" in self.nlp.pipe_names:
            self.ruler = self.nlp.add_pipe(
                "entity_ruler",
                before="ner",
                config=config
            )
        else:
            self.ruler = self.nlp.add_pipe(
                "entity_ruler",
                config=config
            )


    def set_patterns(self, patterns: list[dict]) -> None:
        self._build_empty_ruler()
        if patterns:
            self.ruler.add_patterns(patterns)

    def load_patterns_from_db(self, db_manager) -> list[dict]:
        entities = db_manager.list_entities()
        patterns = []

        for entity in entities:
            name = entity["name"].strip()
            label = entity["category"].strip()

            if not name or not label:
                continue

            patterns.append({
                "label": label,
                "pattern": name
            })

        self.set_patterns(patterns)
        return patterns

    def extract(self, text: str) -> List[EntitySpan]:
        doc = self.nlp(text)
        result = []

        for ent in doc.ents:
            result.append(
                EntitySpan(
                    text=ent.text,
                    label=ent.label_,
                    start=ent.start_char,
                    end=ent.end_char,
                    score=None,
                )
            )

        return result