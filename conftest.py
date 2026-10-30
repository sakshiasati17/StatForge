import sys
from pathlib import Path

# Ensure project root is in sys.path regardless of working directory
root = Path(__file__).resolve().parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))
