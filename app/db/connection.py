import sqlite3
from pathlib import Path
from typing import Optional


def ensure_parent_dir(db_path: str) -> None:
    """
    brief: Ensure the parent directory of the database file exists
    param[in] db_path: Path to the database file 
    """
    if db_path == ":memory:":
        return
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)


def create_connection(db_path: str) -> sqlite3.Connection:
    """
    brief: Create a new SQLite connection to the specified database file
    param[in] db_path: Path to the database file
    return[out] A sqlite3.Connection object with appropriate settings
    """
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    return conn


class ConnectionHolder:
    """
    Helper class to manage the SQLite connection lifecycle.
     - Lazily initializes the connection on first access
     - Ensures parent directory exists for file-based databases
     - Provides a method to close the connection when done
     - Uses WAL journal mode for better concurrency
     - Enables foreign key constraints
     - Sets row_factory to sqlite3.Row for dict-like access to query results
     - Supports in-memory databases with ":memory:" special path
     - Designed to be used within DatabaseManager for managing the connection
    """

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._conn: Optional[sqlite3.Connection] = None

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            ensure_parent_dir(self.db_path)
            self._conn = create_connection(self.db_path)
        return self._conn

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

