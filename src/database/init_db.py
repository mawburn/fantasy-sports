"""Database initialization script."""

import logging

from ..config.settings import settings
from .connection import engine
from .models import Base

logger = logging.getLogger(__name__)


def create_database():
    """Create database and all tables."""
    try:
        # Ensure data directory exists
        data_dir = settings.data_dir / "database"
        data_dir.mkdir(parents=True, exist_ok=True)

        # Create all tables
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created successfully")

    except Exception:
        logger.exception("Failed to create database")
        raise


def drop_database():
    """Drop all database tables."""
    try:
        Base.metadata.drop_all(bind=engine)
        logger.info("Database tables dropped successfully")

    except Exception:
        logger.exception("Failed to drop database")
        raise


def reset_database():
    """Reset database by dropping and recreating all tables."""
    logger.info("Resetting database...")
    drop_database()
    create_database()
    logger.info("Database reset complete")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    create_database()
