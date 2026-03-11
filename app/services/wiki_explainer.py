import re
import wikipediaapi


class WikipediaExplainer:
    def __init__(self, language: str = "en"):
        self.wiki = wikipediaapi.Wikipedia(
            user_agent="ml-lab-2-ner/1.0 (your_email@example.com)",
            language=language,
        )

    def _first_sentence(self, text: str) -> str:
        text = (text or "").strip()
        if not text:
            return ""
        parts = re.split(r'(?<=[.!?])\s+', text, maxsplit=1)
        return parts[0].strip()

    def explain(self, title: str) -> dict:
        page = self.wiki.page(title)

        if not page.exists():
            return {
                "found": False,
                "title": title,
                "summary": None,
                "url": None,
                "source": "wikipedia",
            }

        short_summary = self._first_sentence(page.summary)

        return {
            "found": bool(short_summary),
            "title": page.title,
            "summary": short_summary,
            "url": page.fullurl,
            "source": "wikipedia",
        }
