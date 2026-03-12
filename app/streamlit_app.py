from pathlib import Path

import streamlit as st

from app.db import DatabaseManager
from app.ner.hf_transformers_extractor import HFTokenClassificationExtractor
from app.services.kb_service import KnowledgeBaseService
from app.visualization.ner_viz import NERVisualizer as CategoryVisualizer
from visualization import NERVisualizer as GlobalVisualizer


@st.cache_resource(show_spinner=False)
def get_services(db_path: str):
    db = DatabaseManager(db_path)
    # Load your fine-tuned model exported from the notebook.
    # Put the folder (after unzip) into `models/my_ner_model/` by default.
    extractor = HFTokenClassificationExtractor(model_dir="models/my_ner_model")
    kb = KnowledgeBaseService(db=db, extractor=extractor)
    cat_viz = CategoryVisualizer(extractor=extractor, db=db)
    glob_viz = GlobalVisualizer(db_manager=db)
    return db, extractor, kb, cat_viz, glob_viz


def main():
    st.set_page_config(page_title="NER system visualizer", layout="wide")

    st.title("NER system visualizer")

    db_path = str(Path("data/ner_kb.db"))
    db, extractor, kb, cat_viz, glob_viz = get_services(db_path)

    tab_main, tab_manage, tab_viz = st.tabs(["Dashboard", "Knowledge Base", "Visualization"])

    with tab_main:
        render_main_tab(db=db, kb=kb)

    with tab_manage:
        render_kb_management_tab(db=db, kb=kb)

    with tab_viz:
        render_visualizations_tab(kb=kb, cat_viz=cat_viz)


def render_main_tab(db: DatabaseManager, kb: KnowledgeBaseService):
    st.subheader("Overview")
    stats = kb.stats()
    cols = st.columns(3)
    cols[0].metric("Categories", stats.get("categories", 0))
    cols[1].metric("Entities", stats.get("entities", 0))
    cols[2].metric("Texts", stats.get("texts", 0))

    st.markdown("---")
    st.subheader("Categories")
    cats = db.list_categories()
    if not cats:
        st.info("No categories yet.")
    else:
        st.dataframe(cats, width='stretch', hide_index=True)

    st.markdown("---")
    render_text_parsing_block(kb=kb)


def render_kb_management_tab(db: DatabaseManager, kb: KnowledgeBaseService):
    st.subheader("Knowledge Base management")

    cats = db.list_categories()
    cat_options = [c["name"] for c in cats] if cats else []

    st.markdown("---")
    st.subheader("Entities")

    with st.form("kb_add_entity_form", clear_on_submit=True):
        col1, col2 = st.columns(2)
        with col1:
            ent_name = st.text_input("Entity name", key="kb_ent_name")
            auto_detect = st.checkbox(
                "Auto-detect category with NER model",
                value=False,
                key="kb_ent_auto_detect_cat",
            )
            ent_cat = st.selectbox(
                "Category (manual)",
                options=cat_options,
                key="kb_ent_cat",
                disabled=auto_detect,
            )
        with col2:
            ent_desc = st.text_area(
                "Description (optional)", height=90, key="kb_ent_desc"
            )
        submitted_ent = st.form_submit_button("Add entity")
        if submitted_ent:
            if not ent_name:
                st.warning("Please provide an entity name.")
            else:
                detected_cat = None
                if auto_detect:
                    try:
                        extracted = kb.extractor.extract(ent_name)
                    except Exception:
                        extracted = []

                    # Choose the best matching prediction for the provided surface form.
                    # If extractor returns spans, prefer the longest span; otherwise take the first.
                    best = None
                    for e in extracted or []:
                        if best is None:
                            best = e
                            continue
                        if getattr(e, "end", 0) - getattr(e, "start", 0) > getattr(best, "end", 0) - getattr(best, "start", 0):
                            best = e
                    if best is not None:
                        detected_cat = getattr(best, "label", None) or getattr(best, "label_", None)

                category_to_use = detected_cat or ent_cat
                if not category_to_use:
                    st.warning("Please choose a category (or enable auto-detect).")
                else:
                    # Ensure category exists (auto-create if needed)
                    if not db.get_category(category_to_use):
                        kb.add_category(category_to_use, description="Auto-created from NER model prediction")

                    kb.add_entity_manual(ent_name, category_to_use, ent_desc)
                    if detected_cat:
                        st.success(
                            f"Entity '{ent_name}' added to detected category '{category_to_use}'."
                        )
                    else:
                        st.success(f"Entity '{ent_name}' added to category '{category_to_use}'.")

    st.markdown("---")
    st.subheader("Delete entity")
    all_entities = db.list_entities()
    if not all_entities:
        st.info("No entities yet.")
    else:
        del_name = st.text_input("Entity name to delete", key="kb_del_name")
        if st.button("Delete entity", key="kb_del_btn"):
            if not del_name:
                st.warning("Please enter an entity name.")
            else:
                ok = kb.delete_entity(del_name)
                if ok:
                    st.success(f"Entity '{del_name}' deleted.")
                else:
                    st.warning("Entity not found.")

        st.markdown("---")
        st.subheader("Reassign entity to another category")
        col1, col2 = st.columns(2)
        with col1:
            ent_to_move = st.selectbox(
                "Entity",
                options=[e["name"] for e in all_entities],
                key="kb_move_entity",
            )
        with col2:
            target_cat = st.selectbox(
                "Target category",
                options=cat_options,
                key="kb_move_cat",
            )
        if st.button("Reassign", key="kb_move_btn"):
            if not ent_to_move or not target_cat:
                st.warning("Please choose an entity and a target category.")
            else:
                db.reassign_entity(ent_to_move, target_cat)
                kb.refresh_extractor()
                st.success(f"Entity '{ent_to_move}' reassigned to '{target_cat}'.")

    st.markdown("---")
    st.subheader("Entity search")
    query = st.text_input("Search query", key="kb_entity_search_query")
    if st.button("Search", key="kb_entity_search_btn"):
        st.session_state["kb_entity_search_last_query"] = query

    query = st.session_state.get("kb_entity_search_last_query", query).strip() if query is not None else ""
    if query:
        overview = kb.get_entity_overview(query)
        if not overview["found"]:
            st.info(overview["message"])
        else:
            entity = overview["entity"]
            entity_name = entity["name"]
            st.markdown(f"**Entity:** {entity_name}")
            st.markdown(f"**Category:** {entity['category']}")

            with st.expander("Meaning / explanation", expanded=False):
                explain_term = st.text_input(
                    "Word to explain",
                    value=entity_name,
                    key="kb_explain_query",
                )
                if st.button("Get explanation", key="kb_explain_btn"):
                    info = kb.get_explanation(explain_term)
                    if not info["found"]:
                        st.warning(info.get("message", "Explanation not found"))
                    else:
                        st.success(f"Source: {info['source']}")
                        st.markdown(f"**{info['entity']}** — {info['explanation']}")
                        if info.get("wiki_url"):
                            st.markdown(f"[Wikipedia link]({info['wiki_url']})")

            with st.expander("Related texts (subject/object)", expanded=False):
                from visualization import NERVisualizer as SimpleViz

                viz = SimpleViz(kb.db)
                subj_texts = kb.get_texts_for_entity(entity_name, role="subject")
                obj_texts = kb.get_texts_for_entity(entity_name, role="object")

                st.markdown(f"**Subject:** {len(subj_texts)}")
                with st.expander("Show subject texts", expanded=False):
                    if not subj_texts:
                        st.info("No subject texts found.")
                    for i, t in enumerate(subj_texts, start=1):
                        highlighted_html = viz._highlight_html(t["content"])
                        st.markdown(f"**{i}.**", unsafe_allow_html=False)
                        st.markdown(highlighted_html, unsafe_allow_html=True)

                st.markdown(f"**Object:** {len(obj_texts)}")
                with st.expander("Show object texts", expanded=False):
                    if not obj_texts:
                        st.info("No object texts found.")
                    for i, t in enumerate(obj_texts, start=1):
                        highlighted_html = viz._highlight_html(t["content"])
                        st.markdown(f"**{i}.**", unsafe_allow_html=False)
                        st.markdown(highlighted_html, unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("NER categories")
    with st.form("kb_add_category_form", clear_on_submit=True):
        name = st.text_input("Category name (LABEL)", key="kb_cat_name")
        desc = st.text_input("Description (optional)", key="kb_cat_desc")
        auto_reassign = st.checkbox(
            "Auto-reassign existing entities to this category using rules",
            value=False,
            key="kb_cat_auto_reassign",
        )
        submitted = st.form_submit_button("Add category")
        if submitted:
            if not name:
                st.warning("Please enter a category name.")
            else:
                kb.add_category(name, desc or "")
                if auto_reassign:
                    with st.spinner("Reassigning entities based on local descriptions and rules..."):
                        stats = kb.build_category_from_local_descriptions(
                            category_name=name,
                            source_categories=None,
                            limit=None,
                            fetch_missing_descriptions=True,
                        )
                    st.success(
                        f"Category '{name}' added. "
                        f"Processed {stats.get('processed_entities', 0)} entities, "
                        f"reassigned {stats.get('reassigned_entities', 0)}."
                    )
                else:
                    st.success(f"Category '{name}' added (or already existed).")


def render_text_parsing_block(kb: KnowledgeBaseService):
    st.subheader("Text parsing & highlighting")

    with st.form("ingest_form", clear_on_submit=False):
        text = st.text_area("Text to analyze", height=220, key="main_ingest_text")
        auto_add = st.checkbox(
            "Auto-create unknown categories and entities",
            value=True,
            key="main_ingest_auto_add",
        )
        submitted = st.form_submit_button("Analyze & save")

    if submitted and text:
        result = kb.ingest_text(text=text, source="", auto_add_unknown=auto_add)
        if not result["text_id"]:
            st.warning(result["message"])
            return
        st.success(result["message"])

        from visualization import NERVisualizer as SimpleViz

        viz = SimpleViz(kb.db)
        # Highlight only entities returned by the model (same as in Visualization tab)
        if result["entities"]:
            highlighted_html = viz.highlight_text_entities(text, result["entities"])
        else:
            highlighted_html = viz.highlight_text_entities(text, [])
        st.markdown("**Highlighted entities in text (model output):**", unsafe_allow_html=False)
        st.markdown(highlighted_html, unsafe_allow_html=True)

        # Tables: new entities vs already existed
        new_entities = result.get("new_entities") or []
        new_set = {(e["name"], e["category"]) for e in new_entities}
        already_existed = [
            {"name": e["name"], "category": e["category"], "role": e.get("role", "")}
            for e in result["entities"]
            if (e["name"], e["category"]) not in new_set
        ]

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("New entities")
            if not new_entities:
                st.info("No new entities.")
            else:
                st.dataframe(
                    [{"name": e["name"], "category": e["category"]} for e in new_entities],
                    width='stretch',
                    hide_index=True,
                )
        with col2:
            st.subheader("Already existed")
            if not already_existed:
                st.info("No entities already in KB.")
            else:
                st.dataframe(already_existed, width='stretch', hide_index=True)


def render_visualizations_tab(kb: KnowledgeBaseService, cat_viz: CategoryVisualizer):
    st.subheader("Word clouds for categories")
    cats = kb.db.list_categories()
    cat_names = [c["name"] for c in cats if c["entity_count"] > 0]
    if not cat_names:
        st.info("No categories with entities available for a word cloud.")
    else:
        selected_cat = st.selectbox("Category", options=cat_names)
        if st.button("Generate category word cloud"):
            img_path = cat_viz.make_category_wordcloud(selected_cat)
            st.image(str(img_path), caption=f"Wordcloud: {selected_cat}")

    st.markdown("---")
    st.subheader("Entity highlighting for a random text from the Knowledge Base")
    texts = kb.db.list_texts()
    if not texts:
        st.info("No texts in the Knowledge Base yet.")
        return

    if st.button("Pick a random text", key="viz_random_text_btn"):
        import random

        st.session_state["viz_random_text_id"] = random.choice(texts)["id"]

    text_id = st.session_state.get("viz_random_text_id")
    if text_id is None:
        text_id = texts[0]["id"]
        st.session_state["viz_random_text_id"] = text_id

    text_obj = kb.db.get_text(text_id)
    if not text_obj:
        st.warning("Failed to load the text.")
        return

    from visualization import NERVisualizer as SimpleViz

    viz = SimpleViz(kb.db)
    entities = kb.db.get_entities_in_text(text_id)
    highlighted_html = viz.highlight_text_entities(text_obj["content"], entities)

    st.markdown("**Entities linked to this text (from Knowledge Base):**", unsafe_allow_html=False)
    if not entities:
        st.info("No entities are linked to this text in the Knowledge Base.")
    else:
        st.dataframe(
            [
                {
                    "name": e["name"],
                    "category": e["category"],
                    "role": e.get("role", ""),
                }
                for e in entities
            ],
            width="stretch",
            hide_index=True,
        )

    st.markdown("**Highlighted entities in text (by KB links):**", unsafe_allow_html=False)
    st.markdown(highlighted_html, unsafe_allow_html=True)


if __name__ == "__main__":
    main()

