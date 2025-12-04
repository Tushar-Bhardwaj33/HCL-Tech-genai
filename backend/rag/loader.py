import uuid
from typing import List, Optional
from langchain_chroma import Chroma
from langchain.storage import InMemoryStore
from langchain.schema.document import Document
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langchain.retrievers.multi_vector import MultiVectorRetriever
import os
from dotenv import load_dotenv

# Load the .env file
load_dotenv()

class MultiModalRAGLoader:
    def __init__(self, collection_name="multi_modal_rag", model="nvidia/nv-embedqa-mistral-7b-v2"):
        self.id_key = "doc_id"
        self.vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=NVIDIAEmbeddings(model=model)
        )
        self.docstore = InMemoryStore()
        self.retriever = MultiVectorRetriever(
            vectorstore=self.vectorstore,
            docstore=self.docstore,
            id_key=self.id_key,
        )

    def _generate_ids(self, items: List) -> List[str]:
        return [str(uuid.uuid4()) for _ in items]

    def _create_documents(self, summaries: List[str], ids: List[str]) -> List[Document]:
        return [
            Document(page_content=summary, metadata={self.id_key: doc_id})
            for summary, doc_id in zip(summaries, ids)
        ]

    def add_texts(self, texts: List[str], summaries: List[str]):
        text_ids = self._generate_ids(texts)
        text_docs = self._create_documents(summaries, text_ids)
        self.retriever.vectorstore.add_documents(text_docs)
        self.retriever.docstore.mset(list(zip(text_ids, texts)))

    def add_tables(self, tables: List[str], summaries: List[str]):
        table_ids = self._generate_ids(tables)
        table_docs = self._create_documents(summaries, table_ids)
        self.retriever.vectorstore.add_documents(table_docs)
        self.retriever.docstore.mset(list(zip(table_ids, tables)))

    def add_images(self, images: List[str], summaries: List[str]):
        image_ids = self._generate_ids(images)
        image_docs = self._create_documents(summaries, image_ids)
        self.retriever.vectorstore.add_documents(image_docs)
        self.retriever.docstore.mset(list(zip(image_ids, images)))

    def add_data(
        self,
        texts: Optional[List[str]] = None,
        text_summaries: Optional[List[str]] = None,
        tables: Optional[List[str]] = None,
        table_summaries: Optional[List[str]] = None,
        images: Optional[List[str]] = None,
        image_summaries: Optional[List[str]] = None,
    ):
        if texts and text_summaries:
            self.add_texts(texts, text_summaries)
        if tables and table_summaries:
            self.add_tables(tables, table_summaries)
        if images and image_summaries:
            self.add_images(images, image_summaries)

    def get_retriever(self) -> MultiVectorRetriever:
        return self.retriever