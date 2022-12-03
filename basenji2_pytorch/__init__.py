from pathlib import Path
from .model import Basenji2, PLBasenji2

_lib_dir = Path(__file__).parent
_repo_dir = _lib_dir.parent

params = _repo_dir/'params_human.json'
weights = _repo_dir/'data'/'basenji2.pth'