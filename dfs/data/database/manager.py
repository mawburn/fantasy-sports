"""Database connection and management."""

import sqlite3
import threading
from contextlib import contextmanager
from typing import Optional, Any, List, Dict
from pathlib import Path

from dfs.core.config import config
from dfs.core.logging import get_logger
from dfs.core.exceptions import DataError

logger = get_logger("data.database")


class DatabaseManager:
    """Thread-safe database connection manager."""
    
    def __init__(self, db_path: str = None):
        self.db_path = db_path or config.database.path
        self._local = threading.local()
        
    def get_connection(self) -> sqlite3.Connection:
        """Get thread-local database connection."""
        if not hasattr(self._local, 'connection'):
            try:
                self._local.connection = sqlite3.connect(
                    self.db_path,
                    timeout=config.database.timeout,
                    check_same_thread=False
                )
                self._local.connection.row_factory = sqlite3.Row
                # Enable foreign keys
                self._local.connection.execute("PRAGMA foreign_keys = ON")
            except sqlite3.Error as e:
                raise DataError(f"Failed to connect to database: {e}")
                
        return self._local.connection
    
    def close_connection(self):
        """Close thread-local connection."""
        if hasattr(self._local, 'connection'):
            self._local.connection.close()
            del self._local.connection
    
    @contextmanager
    def transaction(self):
        """Context manager for database transactions."""
        conn = self.get_connection()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
    
    def execute_query(
        self, 
        query: str, 
        params: Optional[tuple] = None
    ) -> sqlite3.Cursor:
        """Execute a query and return cursor."""
        conn = self.get_connection()
        try:
            return conn.execute(query, params or ())
        except sqlite3.Error as e:
            logger.error(f"Query failed: {query[:100]}... Error: {e}")
            raise DataError(f"Database query failed: {e}")
    
    def fetch_one(
        self, 
        query: str, 
        params: Optional[tuple] = None
    ) -> Optional[sqlite3.Row]:
        """Execute query and fetch one result."""
        cursor = self.execute_query(query, params)
        return cursor.fetchone()
    
    def fetch_all(
        self, 
        query: str, 
        params: Optional[tuple] = None
    ) -> List[sqlite3.Row]:
        """Execute query and fetch all results."""
        cursor = self.execute_query(query, params)
        return cursor.fetchall()


# Global database manager instance
_db_manager = None
_db_lock = threading.Lock()


def get_db_manager() -> DatabaseManager:
    """Get global database manager instance."""
    global _db_manager
    if _db_manager is None:
        with _db_lock:
            if _db_manager is None:
                _db_manager = DatabaseManager()
    return _db_manager


def get_db_connection(db_path: str = None) -> sqlite3.Connection:
    """Get database connection (backward compatibility)."""
    if db_path and db_path != config.database.path:
        # Create temporary manager for different path
        temp_manager = DatabaseManager(db_path)
        return temp_manager.get_connection()
    
    return get_db_manager().get_connection()


# Convenience functions for backward compatibility
def execute_query(query: str, params: Optional[tuple] = None) -> sqlite3.Cursor:
    """Execute a query and return cursor."""
    return get_db_manager().execute_query(query, params)


def fetch_one(query: str, params: Optional[tuple] = None) -> Optional[sqlite3.Row]:
    """Execute query and fetch one result."""
    return get_db_manager().fetch_one(query, params)


def fetch_all(query: str, params: Optional[tuple] = None) -> List[sqlite3.Row]:
    """Execute query and fetch all results."""
    return get_db_manager().fetch_all(query, params)