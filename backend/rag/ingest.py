import os
from .chunking.get import GET
from .chunking.pdf_chunker import PDFChunker
from .chunking.text_chunker import TextChunker
from .summarizer import SummarizerAndImageDescriber
from .loader import MultiModalRAGLoader
import json

class INGEST:
    def __init__(self, collection_name, groq_api_key=None, nvidia_api_key=None):
        """Initialize the INGEST instance with collection name and optional API keys."""
        self.collection_name = collection_name
        self.groq_api_key = groq_api_key or os.getenv("GROQ_API_KEY")
        self.nvidia_api_key = nvidia_api_key or os.getenv("NVIDIA_API_KEY")

        # Initialize tools
        self.get_tool = GET()
        self.pdf_chunker_tool = PDFChunker()
        self.text_chunker_tool = TextChunker()
        self.summarizer_tool = SummarizerAndImageDescriber(
            groq_api_key=self.groq_api_key,
            nvidia_api_key=self.nvidia_api_key
        )
        self.loader = MultiModalRAGLoader(collection_name=self.collection_name)

    def process_pdf(self, file_path):
        """Process a PDF file: extract chunks, images, texts, tables, summarize, and return the retriever."""
        try:
            # Step 1: Chunk the PDF
            chunks = self.pdf_chunker_tool.chunk_pdf(file_path)
            if not chunks:
                return f"No chunks extracted from the PDF: {file_path}"

            # Step 2: Extract images, texts, and tables
            try:
                images = self.get_tool.get_images(chunks)
                texts = self.get_tool.get_texts(chunks)
                tables = self.get_tool.get_tables(chunks)
            except Exception as e:
                return f"Error extracting data from chunks: {str(e)}"

            # Step 3: Summarize the extracted content
            try:
                text_summaries = self.summarizer_tool.summarize_texts(texts)
                table_summaries = self.summarizer_tool.summarize_tables(tables)
                image_summaries = self.summarizer_tool.describe_images(images)
            except Exception as e:
                return f"Error summarizing content: {str(e)}"

            # Step 4: Add data to the loader
            self.loader.add_data(
                texts=texts,
                text_summaries=text_summaries,
                tables=tables,
                table_summaries=table_summaries,
                images=images,
                image_summaries=image_summaries
            )

            # Step 5: Return the retriever
            retriever = self.loader.get_retriever()
            if not retriever:
                return "Failed to create retriever."

            return retriever
        
        except Exception as e:
            return f"Unexpected error: {str(e)}"
        
    def process_text(self, text: str):
        """Process a transcript: chunk, summarize, and return retriever."""
        try:
            # Step 1: Chunk the transcript
            chunks = self.text_chunker_tool.chunk_text(text)
            if not chunks:
                return "No chunks extracted from the transcript."

            texts = [chunk.text for chunk in chunks]

            # Step 2: Summarize the text chunks
            try:
                text_summaries = self.summarizer_tool.summarize_texts(texts)
            except Exception as e:
                return f"Error summarizing content: {str(e)}"

            # Step 3: Add data to the loader
            self.loader.add_data(
                texts=texts,
                text_summaries=text_summaries
            )

            # Step 4: Return the retriever
            retriever = self.loader.get_retriever()
            if not retriever:
                return "Failed to create retriever."

            return retriever

        except Exception as e:
            return f"Unexpected error: {str(e)}"