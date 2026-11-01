import sys
from pathlib import Path

# Allows importing schemas, clients and manager from the project folder
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
