from typing import Optional

import sqlite3

from .connection import ConnectionHolder
from .schema import init_schema


class DatabaseManager:
    """
    brief: Provides CRUD operations for categories, entities, texts, and their links
    """

    def __init__(self, db_path: str = ":memory:"):
        self.db_path = db_path
        self._holder = ConnectionHolder(db_path)
        init_schema(self.conn)

    # Connection management

    @property
    def conn(self) -> sqlite3.Connection:
        return self._holder.conn

    def close(self) -> None:
        self._holder.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    # CRUD for categories, entities, texts, links

    def add_category(
        self,
        name: str,
        description: str = "",
    ) -> int:
        """
        brief: Add a category (NER tag). Returns its id. If already exists, returns existing id

        param[in] name: Tag name (unique), e.g. PERSON, ORGANIZATION
        param[in] description: Optional description or explanation for the tag
        return[out] ID of the created or existing category
        """
        cur = self.conn.execute(
            "INSERT OR IGNORE INTO categories(name, description) VALUES (?,?)",
            (name, description),
        )
        self.conn.commit()
        if cur.lastrowid:
            return cur.lastrowid
        # already exists, fetch its id
        return self.get_category(name)["id"]

    def get_category(self, name: str) -> Optional[dict]:
        '''
        brief: Get category by name
        '''
        row = self.conn.execute(
            "SELECT * FROM categories WHERE name = ?", (name,)
        ).fetchone()
        return dict(row) if row else None

    def list_categories(self) -> list[dict]:
        '''
        brief: List all categories with entity counts
        '''
        rows = self.conn.execute(
            "SELECT c.*, COUNT(e.id) AS entity_count "
            "FROM categories c LEFT JOIN entities e ON e.category_id = c.id "
            "GROUP BY c.id ORDER BY c.name"
        ).fetchall()
        return [dict(r) for r in rows]

    def delete_category(self, name: str) -> bool:
        """
        brief:Delete a category. All entities in it will be deleted cascading
        param[in] name: Category name
        return[out] True if a category was deleted, False if not found
        """
        cur = self.conn.execute(
            "DELETE FROM categories WHERE name = ?", (name,)
        )
        self.conn.commit()
        return cur.rowcount > 0

    def update_category(self, name: str, **kwargs) -> bool:
        """
        brief: Update category's description
        param[in] name: Category name to update
        param[in] kwargs: description="new desc"
            (only this field is allowed)
        return[out] True if category was updated, False if not found or no valid fields
        """
        allowed = {"description"}
        fields = {k: v for k, v in kwargs.items() if k in allowed}
        if not fields:
            return False
        sets = ", ".join(f"{k} = ?" for k in fields)
        cur = self.conn.execute(
            f"UPDATE categories SET {sets} WHERE name = ?",
            (*fields.values(), name),
        )
        self.conn.commit()
        return cur.rowcount > 0


    def add_entity(
        self,
        name: str,
        category_name: str,
        description: str = "",
    ) -> int:
        """
        brief: Add an entity. Returns its id. If already exists, returns existing id
        param[in] name: Entity name (unique within category)
        param[in] category_name: Name of the category this entity belongs to
        param[in] description: Optional text description / explanation
        return[out] ID of the created or existing entity
        """
        cat = self.get_category(category_name)
        if cat is None:
            raise ValueError(f"Категория '{category_name}' не найдена")
        cur = self.conn.execute(
            "INSERT OR IGNORE INTO entities(name, category_id, description) "
            "VALUES (?,?,?)",
            (
                name,
                cat["id"],
                description,
            ),
        )
        self.conn.commit()
        if cur.lastrowid:
            return cur.lastrowid
        return self.get_entity(name, category_name)["id"]

    def get_entity(self, name: str, category_name: str = None) -> Optional[dict]:
        """
        brief: Get entity by name (and optionally category)
        param[in] name: Entity name
        param[in] category_name: Optional category name to narrow search
        return[out] Entity dict with keys: id, name, category, description
        """
        if category_name:
            row = self.conn.execute(
                "SELECT e.*, c.name as category FROM entities e "
                "JOIN categories c ON c.id = e.category_id "
                "WHERE e.name = ? AND c.name = ?",
                (name, category_name),
            ).fetchone()
        else:
            row = self.conn.execute(
                "SELECT e.*, c.name as category FROM entities e "
                "JOIN categories c ON c.id = e.category_id "
                "WHERE e.name = ?",
                (name,),
            ).fetchone()
        if not row:
            return None
        return dict(row)

    def get_entity_by_id(self, entity_id: int) -> Optional[dict]:
        '''
        brief: Get entity by ID
        param[in] entity_id: Entity ID
        return[out] Entity dict with keys: id, name, category, description
        '''
        row = self.conn.execute(
            "SELECT e.*, c.name as category FROM entities e "
            "JOIN categories c ON c.id = e.category_id WHERE e.id = ?",
            (entity_id,),
        ).fetchone()
        return dict(row) if row else None

    def list_entities(self, category_name: str = None) -> list[dict]:
        """
        brief: List entities, optionally filtered by category
        param[in] category_name: Optional category name to filter entities
        return[out] List of entity dicts with keys: id, name, category, description
        """
        if category_name:
            rows = self.conn.execute(
                "SELECT e.*, c.name as category FROM entities e "
                "JOIN categories c ON c.id = e.category_id "
                "WHERE c.name = ? ORDER BY e.name",
                (category_name,),
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT e.*, c.name as category FROM entities e "
                "JOIN categories c ON c.id = e.category_id ORDER BY e.name"
            ).fetchall()
        return [dict(r) for r in rows]

    def delete_entity(self, name: str, category_name: str = None) -> bool:
        """
        brief: Delete an entity by name (and optionally category)
        param[in] name: Entity name
        param[in] category_name: Optional category name to narrow deletion
        return[out] True if an entity was deleted, False if not found
        """
        if category_name:
            cat = self.get_category(category_name)
            if not cat:
                return False
            cur = self.conn.execute(
                "DELETE FROM entities WHERE name = ? AND category_id = ?",
                (name, cat["id"]),
            )
        else:
            cur = self.conn.execute(
                "DELETE FROM entities WHERE name = ?", (name,)
            )
        self.conn.commit()
        return cur.rowcount > 0

    def update_entity(self, name: str, category_name: str = None, **kwargs) -> bool:
        """
        brief: Update an entity's attributes
        param[in] name: Entity name
        param[in] category_name: Optional category name to narrow search
        param[in] **kwargs: Attributes to update (description only)
        return[out] True if an entity was updated, False if not found
        """
        entity = self.get_entity(name, category_name)
        if not entity:
            return False
        allowed = {"description"}
        fields = {k: v for k, v in kwargs.items() if k in allowed}
        if not fields:
            return False
        sets = ", ".join(f"{k} = ?" for k in fields)
        cur = self.conn.execute(
            f"UPDATE entities SET {sets} WHERE id = ?",
            (*fields.values(), entity["id"]),
        )
        self.conn.commit()
        return cur.rowcount > 0

    def reassign_entity(self, entity_name: str, new_category_name: str) -> bool:
        """
        brief: Move an entity to a different category
        param[in] entity_name: Name of the entity to move
        param[in] new_category_name: Name of the target category
        return[out] True if the entity was reassigned, False if not found or category doesn't exist
        """
        cat = self.get_category(new_category_name)
        if not cat:
            raise ValueError(f"Категория '{new_category_name}' не найдена")
        cur = self.conn.execute(
            "UPDATE entities SET category_id = ? WHERE name = ?",
            (cat["id"], entity_name),
        )
        self.conn.commit()
        return cur.rowcount > 0


    def add_text(self, content: str, source: str = "") -> int:
        """
        brief: Add a text snippet. Returns its id. If already exists, returns existing id
        param[in] content: Text content (unique)
        param[in] source: Optional source or context info
        return[out] ID of the created or existing text
        """
        cur = self.conn.execute(
            "INSERT OR IGNORE INTO texts(content, source) VALUES (?,?)",
            (content, source),
        )
        self.conn.commit()
        if cur.lastrowid:
            return cur.lastrowid
        row = self.conn.execute(
            "SELECT id FROM texts WHERE content = ?", (content,)
        ).fetchone()
        return row["id"]

    def get_text(self, text_id: int) -> Optional[dict]:
        """
        brief: Get text by ID
        param[in] text_id: Text ID
        return[out] Text dict with keys: id, content, source, added_at
        """
        row = self.conn.execute(
            "SELECT * FROM texts WHERE id = ?", (text_id,)
        ).fetchone()
        return dict(row) if row else None

    def list_texts(self) -> list[dict]:
        """
        brief: List all texts
        return[out] List of text dicts with keys: id, content, source, added
        """
        rows = self.conn.execute(
            "SELECT * FROM texts ORDER BY added_at DESC"
        ).fetchall()
        return [dict(r) for r in rows]

    def delete_text(self, text_id: int) -> bool:
        """
        brief: Delete a text by ID
        param[in] text_id: Text ID
        return[out] True if a text was deleted, False if not found
        """
        cur = self.conn.execute("DELETE FROM texts WHERE id = ?", (text_id,))
        self.conn.commit()
        return cur.rowcount > 0

    def link_entity_to_text(
        self, entity_id: int, text_id: int, role: str = "mention"
    ) -> bool:
        """
        brief: Link an entity to a text snippet with a specific role
        param[in] entity_id: ID of the entity
        param[in] text_id: ID of the text
        param[in] role: Role of the entity in the text (e.g., "subject")
        return[out] True if the link was created, False if it already exists
        """
        cur = self.conn.execute(
            "INSERT OR IGNORE INTO entity_text_links(entity_id, text_id, role) VALUES (?,?,?)",
            (entity_id, text_id, role),
        )
        self.conn.commit()
        return cur.rowcount > 0

    def get_texts_for_entity(self, entity_name: str, role: str = None) -> list[dict]:
        """
        brief: Get all texts linked to an entity, optionally filtered by role
        param[in] entity_name: Name of the entity
        param[in] role: Optional role to filter links (e.g., "subject")
        return[out] List of text dicts with keys: id, content, source, added_at, role
        """
        base = (
            "SELECT t.*, etl.role FROM texts t "
            "JOIN entity_text_links etl ON etl.text_id = t.id "
            "JOIN entities e ON e.id = etl.entity_id "
            "WHERE e.name = ?"
        )
        if role:
            rows = self.conn.execute(
                base + " AND etl.role = ?", (entity_name, role)
            ).fetchall()
        else:
            rows = self.conn.execute(base, (entity_name,)).fetchall()
        return [dict(r) for r in rows]

    def get_entities_in_text(self, text_id: int) -> list[dict]:
        """
        brief: Get all entities linked to a text snippet
        param[in] text_id: ID of the text
        return[out] List of entity dicts with keys: id, name, category, description, role
        """
        rows = self.conn.execute(
            "SELECT e.*, c.name as category, etl.role "
            "FROM entities e "
            "JOIN categories c ON c.id = e.category_id "
            "JOIN entity_text_links etl ON etl.entity_id = e.id "
            "WHERE etl.text_id = ?",
            (text_id,),
        ).fetchall()
        return [dict(r) for r in rows]


    def search_entity(self, query: str) -> list[dict]:
        """
        brief: Search for entities by name or aliases
        param[in] query: Search query
        return[out] List of entity dicts with keys: id, name, category, description
        """
        q = f"%{query}%"
        rows = self.conn.execute(
            "SELECT e.*, c.name as category FROM entities e "
            "JOIN categories c ON c.id = e.category_id "
            "WHERE e.name LIKE ? OR e.description LIKE ?",
            (q, q),
        ).fetchall()
        return [dict(r) for r in rows]

    def stats(self) -> dict:
        """
        brief: Get basic statistics about the database
        return[out] Dict with counts of categories, entities, texts, and links
        """
        cats = self.conn.execute("SELECT COUNT(*) FROM categories").fetchone()[0]
        ents = self.conn.execute("SELECT COUNT(*) FROM entities").fetchone()[0]
        txts = self.conn.execute("SELECT COUNT(*) FROM texts").fetchone()[0]
        links = self.conn.execute(
            "SELECT COUNT(*) FROM entity_text_links"
        ).fetchone()[0]
        return {
            "categories": cats,
            "entities": ents,
            "texts": txts,
            "links": links,
        }

