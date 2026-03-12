from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from app.ner.base import BaseNERExtractor, EntitySpan


class HFTokenClassificationExtractor(BaseNERExtractor):
    """
    NER extractor - our model
    """

    def __init__(
        self,
        model_dir: str = "models/my_ner_model",
        tokenizer_name_or_path: Optional[str] = None,
        device: int = -1,
        aggregation_strategy: str = "simple",
    ):

        from transformers import (  # type: ignore
            AutoModelForTokenClassification,
            AutoTokenizer,
            pipeline,
        )

        model_path = Path(model_dir)
        model_path_str = str(model_path)

        tok_src = tokenizer_name_or_path or model_path_str
        try:
            tokenizer = AutoTokenizer.from_pretrained(tok_src, use_fast=True)
        except Exception:
            # used bert-base-cased as base checkpoint
            tokenizer = AutoTokenizer.from_pretrained("bert-base-cased", use_fast=True)

        model = AutoModelForTokenClassification.from_pretrained(model_path_str)

        # returns char-level offsets (`start`, `end`) when possible.
        self._pipe = pipeline(
            "token-classification",
            model=model,
            tokenizer=tokenizer,
            device=device,
            aggregation_strategy=aggregation_strategy,
        )

    def extract(self, text: str) -> List[EntitySpan]:
        preds = self._pipe(text)
        result: List[EntitySpan] = []

        for p in preds:
            start = p.get("start", None)
            end = p.get("end", None)
            if start is None or end is None:
                # if offsets are missing, skip 
                continue

            label = (
                p.get("entity_group")
                or p.get("entity")
                or p.get("label")
                or "ENT"
            )
            score = p.get("score")
            result.append(
                EntitySpan(
                    text=text[int(start) : int(end)],
                    label=str(label),
                    start=int(start),
                    end=int(end),
                    score=float(score) if score is not None else None,
                )
            )

        return result

