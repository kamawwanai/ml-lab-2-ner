import re
import html
from typing import Dict, Optional

import colorsys
try:
    import matplotlib.colors as mcolors  # type: ignore
except Exception:  # pragma: no cover
    mcolors = None  # type: ignore

from app.db import DatabaseManager


class NERVisualizer:
    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager
        # name/alias: (entity id, category name, category color)
        self._entity_cache = {}
        self._category_color = {}
        self._build_cache()

    def _get_color_for_category(self, category_name: str) -> str:
        """
        Generate category hex colors by its name hash.
        """
        if category_name not in self._category_color:
            hue = (hash(category_name) % 360) / 360.0
            saturation = 0.8
            value = 0.9
            r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
            if mcolors is not None:
                hex_color = mcolors.to_hex((r, g, b))
            else:
                hex_color = "#{:02x}{:02x}{:02x}".format(
                    int(r * 255), int(g * 255), int(b * 255)
                )
            self._category_color[category_name] = hex_color
        return self._category_color[category_name]

    def _build_cache(self):
        """
        Build cache with all entity names.
        """
        cursor = self.db.conn.execute(
            "SELECT id, name FROM categories",
        )
        categories = {row["id"]: row["name"] for row in cursor.fetchall()}

        cursor = self.db.conn.execute(
            "SELECT id, name, category_id FROM entities",
        )
        for row in cursor.fetchall():
            cat_name = categories.get(row["category_id"])
            if not cat_name:
                continue
            color = self._get_color_for_category(cat_name)
            self._entity_cache[row["name"]] = (row["id"], cat_name, color)

    def get_entity_weights(self) -> Dict[str, float]:
        cursor = self.db.conn.execute(
            "SELECT entity_id, COUNT(*) as cnt FROM entity_text_links GROUP BY entity_id",
        )
        weights_by_id = {row["entity_id"]: row["cnt"] for row in cursor.fetchall()}

        word_weights = {}
        for name, (eid, _, _) in self._entity_cache.items():
            word_weights[name] = weights_by_id.get(eid, 1)
        return word_weights

    def _color_func(self, word, font_size, position, orientation, random_state=None, **kwargs):
        # default to grey
        rgb = (0.7, 0.7, 0.7)
        if word in self._entity_cache:
            _, _, color_hex = self._entity_cache[word]
            try:
                if mcolors is not None:
                    rgb = mcolors.to_rgb(color_hex)
            except ValueError:
                pass
        return rgb

    def generate_wordcloud(self, output_file: Optional[str] = None):
        try:
            import matplotlib.pyplot as plt  # type: ignore
            from wordcloud import WordCloud  # type: ignore
        except Exception as e:  # pragma: no cover
            raise ModuleNotFoundError(
                "Для генерации wordcloud нужны зависимости `matplotlib` и `wordcloud`."
            ) from e

        weights = self.get_entity_weights()
        if not weights:
            print("Could not build wordcloud: no entities detected.")
            return

        wc = WordCloud(
            width=800,
            height=400,
            background_color="white",
            color_func=self._color_func,
            prefer_horizontal=1.0,
            collocations=False,
        ).generate_from_frequencies(weights)

        plt.figure(figsize=(12, 6))
        plt.imshow(wc, interpolation="bilinear")
        plt.axis("off")
        if output_file:
            plt.savefig(output_file, bbox_inches="tight")
            print(f"Wordcloud saved. Path: {output_file}.")
        else:
            plt.show()

    def highlight_entities(self, text: str) -> None:
        """
        Highlights entities in inference text according to its categories.
        """
        html_text = self._highlight_html(text)
        with open("highlighted.html", "w", encoding="utf-8") as f:
            f.write(f"<html><body>{html_text}</body></html>")

    def _highlight_html(self, text: str) -> str:
        """
        HTML‑span entity coloring.
        """
        # longer entities are detected first
        entities = sorted(self._entity_cache.keys(), key=len, reverse=True)
        pattern = "|".join(re.escape(i) for i in entities)
        if not pattern:
            return text

        categories_found = set()

        def replacer(match):
            word = match.group(0)
            # Lookup case-insensitive: pattern matches with IGNORECASE but cache keys may differ in case
            key = next((k for k in self._entity_cache if k.lower() == word.lower()), None)
            if key is None:
                return word
            _, cat_name, cat_color = self._entity_cache[key]
            categories_found.add(cat_name)
            return f'<span style="background-color: {cat_color}40; font-weight: bold;">{word}</span>'

        highlighted = re.sub(pattern, replacer, text, flags=re.IGNORECASE)

        if categories_found:
            sorted_cats = sorted(categories_found)
            category_spans = []
            for cat in sorted_cats:
                color = self._get_color_for_category(cat)
                category_spans.append(
                    f'<span style="background-color: {color}40; font-weight: bold;">{cat}</span>'
                )
            category_line = "  ".join(category_spans)
            highlighted += f"\n\n<hr>\n{category_line}"

        return highlighted

    def highlight_text_entities(self, text: str, entities: list[dict]) -> str:
        """
        Highlight entities in a specific text using only entities that are
        explicitly linked to this text in the DB.

        `entities` is expected to be a list of dicts returned by
        `DatabaseManager.get_entities_in_text`, with at least keys:
        - name
        - category
        """
        if not text:
            return ""
        if not entities:
            return html.escape(text)

        names = sorted({e["name"] for e in entities if e.get("name")}, key=len, reverse=True)
        if not names:
            return html.escape(text)

        pattern = "|".join(re.escape(n) for n in names)

        local_cache: Dict[str, tuple[str, str]] = {}
        for e in entities:
            name = e.get("name")
            cat = e.get("category")
            if not name or not cat:
                continue
            color = self._get_color_for_category(str(cat))
            local_cache[name] = (str(cat), color)

        cats_found = set()

        def replacer(match):
            word = match.group(0)
            key = next((k for k in local_cache.keys() if k.lower() == word.lower()), None)
            if key is None:
                return html.escape(word)
            cat_name, cat_color = local_cache[key]
            cats_found.add(cat_name)
            return f'<span style="background-color: {cat_color}40; font-weight: bold;">{html.escape(word)}</span>'

        highlighted = re.sub(pattern, replacer, text, flags=re.IGNORECASE)

        if cats_found:
            sorted_cats = sorted(cats_found)
            category_spans = []
            for cat in sorted_cats:
                color = self._get_color_for_category(cat)
                category_spans.append(
                    f'<span style="background-color: {color}40; font-weight: bold;">{html.escape(cat)}</span>'
                )
            highlighted += "\n\n<hr>\n" + "  ".join(category_spans)

        return highlighted

    def _normalize_label(self, label: str) -> str:
        raw = (label or "").strip()
        if not raw:
            return ""
        if "-" in raw:
            prefix, rest = raw.split("-", 1)
            if prefix.upper() in {"B", "I"} and rest.strip():
                raw = rest
        return raw.strip().lower()

    def highlight_by_spans(self, text: str, spans) -> str:
        if not text:
            return ""

        norm_spans = []
        for s in spans or []:
            start = getattr(s, "start", None)
            end = getattr(s, "end", None)
            label = getattr(s, "label", None)
            if label is None and isinstance(s, dict):
                label = s.get("category") or s.get("label")
            if start is None and isinstance(s, dict):
                start = s.get("start")
            if end is None and isinstance(s, dict):
                end = s.get("end")

            try:
                start_i = int(start)
                end_i = int(end)
            except Exception:
                continue

            if start_i < 0 or end_i <= start_i or end_i > len(text):
                continue

            cat = self._normalize_label(str(label or ""))
            if not cat:
                continue

            norm_spans.append((start_i, end_i, cat))

        if not norm_spans:
            return html.escape(text)

        norm_spans.sort(key=lambda x: (x[0], -(x[1] - x[0])))

        out = []
        cursor = 0
        cats_found = set()

        for start_i, end_i, cat in norm_spans:
            if start_i < cursor:
                continue

            if cursor < start_i:
                out.append(html.escape(text[cursor:start_i]))

            span_text = text[start_i:end_i]
            color = self._get_color_for_category(cat)
            cats_found.add(cat)
            out.append(
                f'<span style="background-color: {color}40; font-weight: bold;">{html.escape(span_text)}</span>'
            )
            cursor = end_i

        if cursor < len(text):
            out.append(html.escape(text[cursor:]))

        highlighted = "".join(out)

        if cats_found:
            sorted_cats = sorted(cats_found)
            category_spans = []
            for cat in sorted_cats:
                color = self._get_color_for_category(cat)
                category_spans.append(
                    f'<span style="background-color: {color}40; font-weight: bold;">{html.escape(cat)}</span>'
                )
            highlighted += "\n\n<hr>\n" + "  ".join(category_spans)

        return highlighted


if __name__ == "__main__":
    # СЮДА ВСТАВИТЬ БД
    db = DatabaseManager("data/ner_kb.db")
    viz = NERVisualizer(db)

    viz.generate_wordcloud("wordcloud.png")

    sample = input("Paste text sample here:\n").strip()
    viz.highlight_entities(sample)

    db.close()
