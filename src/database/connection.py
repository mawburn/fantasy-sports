"""Database connection and session management using SQLAlchemy.

This module implements the core database connectivity patterns for the application.
It handles:
1. Database engine creation with connection pooling
2. Session factory configuration for ORM operations
3. Multiple session management patterns for different use cases
4. Proper resource cleanup and error handling

Key Concepts for Beginners:

Database Engine: The core interface to the database. Think of it as the
"connection factory" that manages the actual database connections.

Session: A workspace for ORM operations. All database operations (queries,
inserts, updates) happen within a session context.

Connection Pooling: Reuses database connections instead of creating new ones
for each request, improving performance significantly.

Context Managers: Python's "with" statement pattern that ensures proper
resource cleanup even if errors occur.

Dependency Injection: FastAPI's pattern for providing database sessions
to route handlers automatically.

Session Patterns Provided:
1. get_session(): Manual session with automatic commit/rollback
2. get_session_context(): Context manager for with statements
3. get_db(): FastAPI dependency injection pattern
"""

from collections.abc import Generator  # Type hint for generator functions
from contextlib import contextmanager  # Decorator for context manager functions

from sqlalchemy import create_engine  # Database engine factory
from sqlalchemy.orm import Session, sessionmaker  # ORM session management

from ..config.settings import settings  # Application configuration

# Create the database engine - the core interface to our database
# This is created once at module load time and reused throughout the application
engine = create_engine(
    settings.database_url,  # Database connection string (sqlite:///path or postgresql://...)
    echo=settings.database_echo,  # Log all SQL queries (useful for debugging)
    pool_size=settings.database_pool_size,  # Number of connections to maintain in pool
    pool_pre_ping=True,  # Test connections before use (handles disconnects)
)

# Create session factory - a class that produces database sessions
# This factory is configured once and used to create sessions throughout the app
SessionLocal = sessionmaker(
    autocommit=False,  # Require explicit session.commit() for transactions
    autoflush=False,  # Don't automatically flush changes before queries
    bind=engine,  # Associate with our database engine
)

# Why these settings?
# autocommit=False: Gives us explicit control over transactions
# autoflush=False: Prevents unexpected database hits during complex operations
# bind=engine: Links this session factory to our specific database


def get_session() -> Generator[Session, None, None]:
    """Get database session with automatic commit/rollback and cleanup.

    This is the recommended pattern for most database operations.
    It provides:
    1. Automatic session creation
    2. Automatic commit on success
    3. Automatic rollback on errors
    4. Guaranteed session cleanup

    Usage:
        for session in get_session():
            # Do database work
            user = session.query(User).first()
            # Automatically committed and closed

    Error Handling:
    If any exception occurs, the transaction is rolled back and the
    exception is re-raised. This ensures database consistency.
    """
    session = SessionLocal()  # Create new session
    try:
        yield session  # Provide session to caller
        session.commit()  # Commit transaction if no errors
    except Exception:
        session.rollback()  # Undo changes if error occurs
        raise  # Re-raise the exception for caller to handle
    finally:
        session.close()  # Always close session to free connections


@contextmanager
def get_session_context() -> Generator[Session, None, None]:
    """Context manager wrapper for database sessions.

    This provides the same functionality as get_session() but with
    Python's "with" statement syntax for more readable code.

    Usage:
        with get_session_context() as session:
            user = session.query(User).first()
            session.add(new_user)
            # Automatically committed and closed when exiting 'with' block

    Benefits:
    - More Pythonic syntax
    - Clear scope boundaries
    - Explicit resource management
    - Same error handling as get_session()
    """
    with get_session() as session:
        yield session


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency for database sessions.

    This function is designed specifically for use with FastAPI's
    dependency injection system. Unlike get_session(), it:
    - Does NOT automatically commit (lets route handlers control transactions)
    - Does NOT rollback on errors (FastAPI handles HTTP error responses)
    - ONLY ensures session cleanup

    Usage in FastAPI routes:
        @app.get("/users/")
        def get_users(db: Session = Depends(get_db)):
            users = db.query(User).all()
            return users

    FastAPI automatically:
    1. Calls this function to get a session
    2. Injects the session into the route handler
    3. Ensures cleanup after the request completes

    Transaction Control:
    Route handlers must explicitly call db.commit() to save changes.
    This gives fine-grained control over when data is persisted.
    """
    session = SessionLocal()  # Create session for this request
    try:
        yield session  # Provide session to FastAPI route handler
    finally:
        session.close()  # Cleanup session after request completes
