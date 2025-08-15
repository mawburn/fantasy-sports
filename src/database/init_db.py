"""Database initialization script for SQLAlchemy-based applications.

This module provides utilities for setting up and managing the database schema.
It's typically used during:
1. Initial application setup
2. Development environment creation
3. Testing database preparation
4. Deployment pipeline database setup

Database Lifecycle Operations:
- create_database(): Initialize schema from SQLAlchemy models
- drop_database(): Remove all tables (destructive operation)
- reset_database(): Complete refresh (drop + create)

For Beginners:

Database Schema: The structure of your database (tables, columns, indexes).
SQLAlchemy can automatically create this structure from your model definitions.

Metadata: SQLAlchemy's representation of your database schema. It's built
automatically from your model classes (Team, Player, Game, etc.).

Database vs Tables: The database file/connection exists, but tables must be
created separately. This script handles table creation.

Idempotent Operations: create_all() safely handles existing tables,
while drop_all() safely handles non-existing tables.

Directory Management: For file-based databases (SQLite), we must ensure
the directory structure exists before creating the database file.
"""

import logging  # For operation tracking and error reporting

from ..config.settings import settings  # Application configuration
from .connection import engine  # Pre-configured database engine
from .models import Base  # Base class containing all model metadata

# Set up logging for this module to track database operations
logger = logging.getLogger(__name__)


def create_database():
    """Create database schema and all tables from SQLAlchemy models.

    This function:
    1. Ensures the database directory exists (for SQLite)
    2. Creates all tables defined in models.py
    3. Sets up indexes and constraints
    4. Is idempotent - safe to run multiple times

    SQLAlchemy Magic:
    Base.metadata contains the schema definition for all models that
    inherit from Base. create_all() reads this metadata and generates
    the appropriate CREATE TABLE statements.

    For Beginners:

    Path Operations: settings.data_dir / "database" uses Python's pathlib
    to build file paths that work on any operating system.

    mkdir(parents=True, exist_ok=True):
    - parents=True: Create intermediate directories if needed
    - exist_ok=True: Don't error if directory already exists

    Exception Handling: We catch all exceptions, log them with full
    stack traces, then re-raise so the caller knows something failed.
    """
    try:
        # For SQLite databases, ensure the directory structure exists
        # This prevents "directory not found" errors when creating the database file
        data_dir = settings.data_dir / "database"
        data_dir.mkdir(parents=True, exist_ok=True)

        # Create all tables from our model definitions
        # This reads the metadata from Base (which includes all our model classes)
        # and generates CREATE TABLE statements for each one
        Base.metadata.create_all(bind=engine)

        logger.info("Database tables created successfully")

    except Exception:
        # Log the full exception with stack trace for debugging
        logger.exception("Failed to create database")
        raise  # Re-raise so caller knows the operation failed


def drop_database():
    """Drop all database tables - DESTRUCTIVE OPERATION.

    This function:
    1. Removes all tables from the database
    2. Destroys all data permanently
    3. Removes indexes, constraints, and relationships
    4. Is idempotent - safe to run on empty databases

    WARNING: This operation cannot be undone. All data will be lost.

    Use Cases:
    - Resetting development databases
    - Cleaning up test environments
    - Preparing for schema migrations
    - Emergency data purging

    SQLAlchemy Implementation:
    drop_all() generates DROP TABLE statements for all tables
    defined in our models, handling foreign key dependencies
    in the correct order.
    """
    try:
        # Drop all tables defined in our models
        # SQLAlchemy handles the correct order to avoid foreign key constraint errors
        Base.metadata.drop_all(bind=engine)

        logger.info("Database tables dropped successfully")

    except Exception:
        # Log any errors that occur during table dropping
        logger.exception("Failed to drop database")
        raise


def reset_database():
    """Reset database by dropping and recreating all tables.

    This is a complete database refresh that:
    1. Destroys all existing data (drop_database)
    2. Recreates the schema from models (create_database)
    3. Results in a clean, empty database

    Common Use Cases:
    - Development environment resets
    - Test suite preparation
    - Schema change deployment
    - Data corruption recovery

    WARNING: All data will be permanently lost.

    Operation Order:
    We must drop before create to handle schema changes.
    If we only created, existing tables with different schemas
    could cause conflicts.
    """
    logger.info("Resetting database...")

    # First drop all existing tables and data
    drop_database()

    # Then recreate clean schema
    create_database()

    logger.info("Database reset complete")


# Script execution entry point
# This runs when the file is executed directly (python init_db.py)
if __name__ == "__main__":
    # Configure logging to show INFO level messages to console
    # This lets us see the success/failure messages from our functions
    logging.basicConfig(level=logging.INFO)

    # Create the database schema
    # This is the most common operation - setting up a new database
    create_database()
