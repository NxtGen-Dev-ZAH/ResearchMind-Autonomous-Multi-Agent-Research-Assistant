from typing import Dict, List, Optional, Union
from pydantic import BaseModel
import chromadb
from chromadb.config import Settings
import numpy as np
from datetime import datetime

class DocumentMetadata(BaseModel):
    source: str
    timestamp: str
    doc_type: str
    task_id: str
    confidence: float
    tags: List[str] = []

class Document(BaseModel):
    content: str
    metadata: DocumentMetadata
    embedding: Optional[List[float]] = None

class VectorDatabase:
    def __init__(self, persist_directory: str = "./chroma_db"):
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Initialize collections
        self._init_collections()

    def _init_collections(self):
        """Initialize different collections for different types of data"""
        self.research_collection = self.client.get_or_create_collection(
            name="research_results",
            metadata={"hnsw:space": "cosine"}
        )
        
        self.analysis_collection = self.client.get_or_create_collection(
            name="analysis_results",
            metadata={"hnsw:space": "cosine"}
        )
        
        self.synthesis_collection = self.client.get_or_create_collection(
            name="synthesis_results",
            metadata={"hnsw:space": "cosine"}
        )

    async def store(
        self,
        documents: Union[Document, List[Document]],
        collection_name: str = "research_results"
    ) -> List[str]:
        """
        Store documents in the vector database
        """
        if isinstance(documents, Document):
            documents = [documents]

        collection = self.client.get_collection(collection_name)
        
        # Prepare data for storage
        docs = [doc.content for doc in documents]
        metadatas = [doc.metadata.dict() for doc in documents]
        ids = [f"{collection_name}_{i}_{datetime.now().timestamp()}" for i in range(len(documents))]
        
        # Store in database
        collection.add(
            documents=docs,
            metadatas=metadatas,
            ids=ids
        )
        
        return ids

    async def search(
        self,
        query: str,
        collection_name: str = "research_results",
        n_results: int = 5,
        filter_criteria: Optional[Dict] = None
    ) -> List[Document]:
        """
        Search for similar documents
        """
        collection = self.client.get_collection(collection_name)
        
        # Prepare filter
        where_clause = filter_criteria if filter_criteria else {}
        
        # Execute search
        results = collection.query(
            query_texts=[query],
            n_results=n_results,
            where=where_clause
        )
        
        # Convert results to Documents
        documents = []
        for i in range(len(results['documents'][0])):
            doc = Document(
                content=results['documents'][0][i],
                metadata=DocumentMetadata(**results['metadatas'][0][i])
            )
            documents.append(doc)
        
        return documents

    async def update(
        self,
        doc_id: str,
        new_content: str,
        new_metadata: Optional[Dict] = None,
        collection_name: str = "research_results"
    ):
        """
        Update existing document
        """
        collection = self.client.get_collection(collection_name)
        
        update_data = {
            "documents": [new_content],
            "ids": [doc_id]
        }
        
        if new_metadata:
            update_data["metadatas"] = [new_metadata]
        
        collection.update(**update_data)

    async def delete(
        self,
        doc_ids: Union[str, List[str]],
        collection_name: str = "research_results"
    ):
        """
        Delete documents from database
        """
        if isinstance(doc_ids, str):
            doc_ids = [doc_ids]
            
        collection = self.client.get_collection(collection_name)
        collection.delete(ids=doc_ids)

    async def get_similar_by_metadata(
        self,
        metadata_filters: Dict,
        collection_name: str = "research_results",
        n_results: int = 5
    ) -> List[Document]:
        """
        Get similar documents based on metadata filters
        """
        collection = self.client.get_collection(collection_name)
        
        results = collection.query(
            query_texts=[""],  # Empty query to match only on metadata
            n_results=n_results,
            where=metadata_filters
        )
        
        documents = []
        for i in range(len(results['documents'][0])):
            doc = Document(
                content=results['documents'][0][i],
                metadata=DocumentMetadata(**results['metadatas'][0][i])
            )
            documents.append(doc)
        
        return documents