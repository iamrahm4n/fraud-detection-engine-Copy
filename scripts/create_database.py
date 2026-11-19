
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fraud_engine.database.database import engine
from fraud_engine.database.models import Base

# Create database tables
Base.metadata.create_all(bind=engine)

print("Database created successfully!")
print("Location:", PROJECT_ROOT / "data" / "fraud_detection.db")
