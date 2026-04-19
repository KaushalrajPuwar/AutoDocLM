"""
Shared data model for all chunkers (Tiers 1, 2, 3).
All parsers emit Chunk instances, which are serialized to chunks.jsonl.
"""
from dataclasses import dataclass, field, asdict
from typing import List, Optional


@dataclass
class Chunk:
    chunk_id: str              # e.g. "routes.py::UserView.get"
    file: str                  # relative file path
    language: str              # "python", "javascript", "typescript", etc.
    symbol: str                # qualified name, e.g. "UserView.get"
    chunk_type: str            # "function" | "method" | "class" | "block"
    parent_class: Optional[str]       # parent class name if method
    decorators: List[str]             # list of decorator strings, e.g. ["@app.route('/users')"]
    line_start: int            # 1-based start line
    line_end: int              # 1-based end line
    chunk_text: str            # actual source code text (with decorator + class prefix)
    docstring: Optional[str] = None   # extracted docstring if available

    def to_dict(self) -> dict:
        return asdict(self)
