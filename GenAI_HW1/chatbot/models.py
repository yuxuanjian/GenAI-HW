from dataclasses import dataclass, field


@dataclass
class DocumentContext:
    file_name: str = ""
    current_chunk: str = ""
    chunk_idx_display: int = 0
    total_chunks: int = 0
    chunks: list[str] = field(default_factory=list)

    def select_chunk(self, index: int) -> None:
        if not self.chunks:
            self.current_chunk = ""
            self.chunk_idx_display = 0
            return

        self.current_chunk = self.chunks[index]
        self.chunk_idx_display = index + 1


@dataclass
class SidebarSettings:
    selected_model: str
    system_prompt: str
    temperature: float
    max_tokens: int
    document_context: DocumentContext
