from dataclasses import dataclass, field
from typing import Any


@dataclass
class SidebarSettings:
    small_model: str
    large_model: str
    vision_model: str
    system_prompt: str
    temperature: float
    max_tokens: int
    pending_image: Any = None


@dataclass
class RouteDecision:
    selected_model: str
    tools: list[str] = field(default_factory=list)
    use_vision: bool = False
    reason: str = ""


@dataclass
class ToolOutput:
    name: str
    content: str
    metadata: dict = field(default_factory=dict)


@dataclass
class RetrievalHit:
    doc_id: str
    doc_name: str
    chunk_index: int
    content: str
    score: float
