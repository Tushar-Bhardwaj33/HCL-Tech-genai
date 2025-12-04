from unstructured.partition.pdf import partition_pdf
from .get import GET

__all__ = [
    "PDFChunker"
]

"""references: https://docs.unstructured.io/open-source/core-functionality/partitioning"""

class PDFChunker:
    DEFAULT_IMAGE_BLOCK_TYPES = ["Image"]
    DEFAULT_CHUNKING_STRATEGY = "by_title"

    def __init__(
        self,
        output_path: str = "./content/",
        infer_table_structure: bool = True,
        strategy: str = "hi_res",
        extract_image_block_types: list = ["Image"],
        extract_image_block_to_payload: bool = True,
        chunking_strategy: str = "by_title",
        max_characters: int = 10000,
        combine_text_under_n_chars: int = 2000,
        new_after_n_chars: int = 6000
    ):
        self.chunks = None
        self.output_path = output_path
        self.infer_table_structure = infer_table_structure
        self.strategy = strategy
        self.extract_image_block_types = extract_image_block_types
        self.extract_image_block_to_payload = extract_image_block_to_payload
        self.chunking_strategy = chunking_strategy
        self.max_characters = max_characters
        self.combine_text_under_n_chars = combine_text_under_n_chars
        self.new_after_n_chars = new_after_n_chars

    @classmethod
    def get_available_methods(cls):
        return [cls.chunk_pdf, cls.get_texts, cls.get_tables, cls.get_images, cls.get_metadata]

    def chunk_pdf(self, file_name: str):
        if not file_name:
            raise ValueError("file_name cannot be empty.")
        file_path = self.output_path + file_name

        self.chunks = partition_pdf(
            filename=file_path,
            infer_table_structure=self.infer_table_structure,
            strategy=self.strategy,
            extract_image_block_types=self.extract_image_block_types,
            extract_image_block_to_payload=self.extract_image_block_to_payload,
            chunking_strategy=self.chunking_strategy,
            max_characters=self.max_characters,
            combine_text_under_n_chars=self.combine_text_under_n_chars,
            new_after_n_chars=self.new_after_n_chars,
        )
        return self.chunks
    
    def get_images(self):
        return GET.get_images(self.chunks)

    def get_texts(self):
        return GET.get_texts(self.chunks)

    def get_tables(self):
        return GET.get_tables(self.chunks)
    
    def get_metadata(self):
        return GET.get_metadata(self.chunks)