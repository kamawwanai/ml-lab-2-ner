import re
from typing import Dict, Optional

import colorsys
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from wordcloud import WordCloud

from db import DatabaseManager


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
            hex_color = mcolors.to_hex((r, g, b))
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
        """
        Weights = count of entity entries in entity_text_links.
        For no links: weight = 1.
        """
        cursor = self.db.conn.execute(
            "SELECT entity_id, COUNT(*) as cnt FROM entity_text_links GROUP BY entity_id",
        )
        weights_by_id = {row["entity_id"]: row["cnt"] for row in cursor.fetchall()}

        word_weights = {}
        for name, (eid, _, _) in self._entity_cache.items():
            word_weights[name] = weights_by_id.get(eid, 1)
        return word_weights

    def _color_func(self, word, font_size, position, orientation, random_state=None, **kwargs):
        """
        RGB color transformer for wordcloud.
        Unused arguments for wordcloud usage.
        """
        # default to grey
        rgb = (0.7, 0.7, 0.7)
        if word in self._entity_cache:
            _, _, color_hex = self._entity_cache[word]
            try:
                rgb = mcolors.to_rgb(color_hex)
            except ValueError:
                pass
        return rgb

    def generate_wordcloud(self, output_file: Optional[str] = None):
        """
        Generates and saves entity wordcloud with category coloring.
        If output_file is configured, saves the image, otherwise shows it in plt.
        """
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
            _, cat_name, cat_color = self._entity_cache[word]
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


if __name__ == "__main__":
    # СЮДА ВСТАВИТЬ БД
    db = DatabaseManager("йоу")
    viz = NERVisualizer(db)

    viz.generate_wordcloud("wordcloud.png")

    sample = input("Paste text sample here:\n").strip()
    viz.highlight_entities(sample)

    db.close()
