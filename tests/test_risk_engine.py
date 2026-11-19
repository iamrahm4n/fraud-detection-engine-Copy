import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fraud_engine.detection.risk_engine import RiskEngine


engine = RiskEngine()

transactions = [0.08, 0.65, 0.97]

for probability in transactions:
    result = engine.assess_transaction(probability)
    print(result)
