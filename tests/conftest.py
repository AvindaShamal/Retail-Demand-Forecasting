import os
import sys

# Ensure the project root is on sys.path so top-level packages like `data`
# and `src` can be imported when running pytest from any CWD.
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
