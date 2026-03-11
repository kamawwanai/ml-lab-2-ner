import sqlite3


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS categories (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT    NOT NULL UNIQUE,
    description TEXT
);

CREATE TABLE IF NOT EXISTS entities (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT    NOT NULL,
    normalized_name TEXT    NOT NULL,
    category_id INTEGER NOT NULL REFERENCES categories(id) ON DELETE CASCADE,
    description         TEXT,
    description_source  TEXT,
    wiki_url            TEXT,
    description_updated_at TIMESTAMP,
    UNIQUE(name, category_id)
);

CREATE TABLE IF NOT EXISTS texts (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    content    TEXT NOT NULL UNIQUE,
    source     TEXT
    added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS entity_text_links (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    entity_id  INTEGER NOT NULL REFERENCES entities(id)  ON DELETE CASCADE,
    text_id    INTEGER NOT NULL REFERENCES texts(id)     ON DELETE CASCADE,
    role       TEXT    NOT NULL DEFAULT 'mention',
    UNIQUE(entity_id, text_id, role)
);

CREATE INDEX IF NOT EXISTS idx_entities_name     ON entities(name);
CREATE INDEX IF NOT EXISTS idx_entities_norm_name  ON entities(normalized_name);
CREATE INDEX IF NOT EXISTS idx_entities_category ON entities(category_id);
CREATE INDEX IF NOT EXISTS idx_links_entity      ON entity_text_links(entity_id);
CREATE INDEX IF NOT EXISTS idx_links_text        ON entity_text_links(text_id);
"""


def init_schema(conn: sqlite3.Connection) -> None:
    """
    Initialize the database schema.
    """
    statements = SCHEMA_SQL.strip().split(";")
    for stmt in statements:
        sql = stmt.strip()
        if not sql:
            continue
        conn.execute(sql)
    conn.commit()

