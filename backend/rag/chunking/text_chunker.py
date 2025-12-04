from unstructured.partition.text import partition_text

__all__ = [
    "TextChunker"
]

class TextChunker:
    def __init__(
        self,
        chunking_strategy: str = "by_title",
        max_characters: int = 10000,
        combine_text_under_n_chars: int = 2000,
        new_after_n_chars: int = 6000
    ):
        self.chunks = None
        self.chunking_strategy = chunking_strategy
        self.max_characters = max_characters
        self.combine_text_under_n_chars = combine_text_under_n_chars
        self.new_after_n_chars = new_after_n_chars

    def chunk_text(self, text: str):
        if not text:
            raise ValueError("text cannot be empty.")

        self.chunks = partition_text(
            text=text,
            chunking_strategy=self.chunking_strategy,
            max_characters=self.max_characters,
            combine_text_under_n_chars=self.combine_text_under_n_chars,
            new_after_n_chars=self.new_after_n_chars,
        )
        return self.chunks