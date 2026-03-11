from dataclasses import dataclass
from typing import List, Optional


@dataclass
class EntitySpan:
    text: str
    label: str
    start: int
    end: int
    score: Optional[float] = None


class BaseNERExtractor:
    def extract(self, text: str) -> List[EntitySpan]:
        raise NotImplementedError
