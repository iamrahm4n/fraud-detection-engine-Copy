from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Find the main project folder
PROJECT_ROOT = Path(__file__).resolve().parents[3]

# Create the SQLite database file
DATABASE_PATH = PROJECT_ROOT / "data" / "fraud_detection.db"

DATABASE_URL = f"sqlite:///{DATABASE_PATH.as_posix()}"

# Create database connection
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False}
)

# Create database sessions
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)

print("Database connection configured!")
