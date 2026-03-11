from collections import Counter
from pathlib import Path

from spacy import displacy
from wordcloud import WordCloud
import matplotlib.pyplot as plt


class NERVisualizer:
    def __init__(self, extractor, db):
        self.extractor = extractor
        self.db = db

    def render_text_entities_html(self, text: str, output_file: str = "entity_render.html") -> str:
        doc = self.extractor.nlp(text)

        html = displacy.render(
            doc,
            style="ent",
            page=True,
            options={
                "ents": sorted(list({ent.label_ for ent in doc.ents}))
            },
        )

        output_path = Path(output_file)
        output_path.write_text(html, encoding="utf-8")
        return str(output_path)

    def make_category_wordcloud(self, category_name: str, output_file: str = None) -> str:
        entities = self.db.list_entities(category_name=category_name)
        if not entities:
            raise ValueError(f"No entities found for category '{category_name}'")

        freq = Counter()
        for entity in entities:
            texts = self.db.get_texts_for_entity(entity["name"])
            freq[entity["name"]] = max(len(texts), 1)

        wc = WordCloud(
            width=1200,
            height=600,
            background_color="white",
            collocations=False,
        ).generate_from_frequencies(dict(freq))

        if output_file is None:
            safe_name = category_name.lower().replace(" ", "_")
            output_file = f"wordcloud_{safe_name}.png"

        plt.figure(figsize=(12, 6))
        plt.imshow(wc, interpolation="bilinear")
        plt.axis("off")
        plt.tight_layout()

        output_path = Path(output_file)
        plt.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close()

        return str(output_path)

    def make_all_categories_wordclouds(self, output_dir: str = "viz") -> list[str]:
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        paths = []
        for cat in self.db.list_categories():
            if cat["entity_count"] == 0:
                continue
            path = self.make_category_wordcloud(
                category_name=cat["name"],
                output_file=str(Path(output_dir) / f"wordcloud_{cat['name'].lower()}.png"),
            )
            paths.append(path)

        return paths
