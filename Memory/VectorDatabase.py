from typing import Dict, List, Optional, Union, Any
from pydantic import BaseModel, Field
from datetime import datetime
from langchain_community.vectorstores import Pinecone
from langchain_core.documents import Document as LangChainDocument
from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings
import pinecone
import json
import uuid
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class DocumentMetadata(BaseModel):
    """Enhanced metadata for documents"""

    source: str
    timestamp: datetime
    doc_type: str = Field(default="general")  # research, analysis, synthesis
    task_id: Optional[str] = None
    author: Optional[str] = None
    tags: List[str] = []
    confidence: float = 1.0
    additional: Dict[str, Any] = {}


class EnhancedDocument(BaseModel):
    """Document model with enhanced metadata"""

    content: str
    metadata: DocumentMetadata
    embedding: Optional[List[float]] = None


class VectorStore:
    """Enhanced vector store using Pinecone integration"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        environment: Optional[str] = None,
        embedding_model: Optional[Embeddings] = None,
        collection_metadata: Optional[Dict[str, Dict]] = None,
    ):
        """
        Initialize vector store with Pinecone configuration.

        Args:
            api_key: Pinecone API key (defaults to environment variable)
            environment: Pinecone environment (defaults to environment variable)
            embedding_model: Custom embedding model (defaults to OpenAIEmbeddings)
            collection_metadata: Optional metadata for collections
        """
        # Initialize Pinecone
        self.api_key = api_key or os.getenv("PINECONE_API_KEY")
        self.environment = environment or os.getenv("PINECONE_ENVIRONMENT")
        
        if not self.api_key or not self.environment:
            raise ValueError("Pinecone API key and environment must be provided")

        pinecone.init(api_key=self.api_key, environment=self.environment)
        
        self.embedding_model = embedding_model or OpenAIEmbeddings()
        self.collection_metadata = collection_metadata or {
            "research": {"description": "Research documents and findings"},
            "analysis": {"description": "Analysis results and interpretations"},
            "synthesis": {"description": "Synthesized information and conclusions"},
        }
        self.vector_stores: Dict[str, Pinecone] = {}
        self._initialize_collections()

    def _initialize_collections(self) -> None:
        """Initialize or create Pinecone indexes for collections"""
        dimension = 1536  # OpenAI embeddings dimension
        
        for collection_name, metadata in self.collection_metadata.items():
            # Check if index exists
            if collection_name not in pinecone.list_indexes():
                # Create new index
                pinecone.create_index(
                    name=collection_name,
                    dimension=dimension,
                    metadata_config={"indexed": ["doc_type", "source", "task_id", "author"]}
                )

            # Initialize Langchain's Pinecone integration
            index = pinecone.Index(collection_name)
            self.vector_stores[collection_name] = Pinecone(
                index=index,
                embedding=self.embedding_model,
                text_key="text"
            )

    def _convert_to_langchain_document(
        self, doc: EnhancedDocument
    ) -> LangChainDocument:
        """Convert our document model to LangChain document format"""
        metadata = {
            "source": doc.metadata.source,
            "timestamp": doc.metadata.timestamp.isoformat(),
            "doc_type": doc.metadata.doc_type,
            "task_id": doc.metadata.task_id,
            "author": doc.metadata.author,
            "tags": json.dumps(doc.metadata.tags),
            "confidence": doc.metadata.confidence,
            "additional": json.dumps(doc.metadata.additional),
        }
        return LangChainDocument(page_content=doc.content, metadata=metadata)

    def _convert_from_langchain_document(
        self, doc: LangChainDocument
    ) -> EnhancedDocument:
        """Convert LangChain document back to our document model"""
        metadata = DocumentMetadata(
            source=doc.metadata["source"],
            timestamp=datetime.fromisoformat(doc.metadata["timestamp"]),
            doc_type=doc.metadata["doc_type"],
            task_id=doc.metadata["task_id"],
            author=doc.metadata["author"],
            tags=json.loads(doc.metadata["tags"]),
            confidence=doc.metadata["confidence"],
            additional=json.loads(doc.metadata["additional"])
        )
        return EnhancedDocument(content=doc.page_content, metadata=metadata)

    async def store(
        self,
        documents: Union[EnhancedDocument, List[EnhancedDocument]],
        collection_name: str = "research",
    ) -> List[str]:
        """
        Store documents in Pinecone

        Args:
            documents: Single document or list of documents to store
            collection_name: Name of collection to store in

        Returns:
            List of document IDs
        """
        if isinstance(documents, EnhancedDocument):
            documents = [documents]

        if collection_name not in self.vector_stores:
            raise ValueError(f"Collection {collection_name} does not exist")

        langchain_docs = [self._convert_to_langchain_document(doc) for doc in documents]
        ids = [str(uuid.uuid4()) for _ in documents]

        self.vector_stores[collection_name].add_documents(
            documents=langchain_docs,
            ids=ids
        )

        return ids

    async def search(
        self,
        query: str,
        collection_name: str = "research",
        n_results: int = 5,
        filter_criteria: Optional[Dict] = None,
        include_metadata: bool = True,
    ) -> List[EnhancedDocument]:
        """
        Search for similar documents in Pinecone

        Args:
            query: Search query
            collection_name: Name of collection to search
            n_results: Number of results to return
            filter_criteria: Optional metadata filters
            include_metadata: Whether to include metadata in results

        Returns:
            List of matching documents
        """
        if collection_name not in self.vector_stores:
            raise ValueError(f"Collection {collection_name} does not exist")

        results = self.vector_stores[collection_name].similarity_search(
            query=query,
            k=n_results,
            filter=filter_criteria
        )

        return [self._convert_from_langchain_document(doc) for doc in results]

    async def update(
        self, doc_id: str, document: EnhancedDocument, collection_name: str = "research"
    ) -> None:
        """
        Update an existing document in Pinecone

        Args:
            doc_id: ID of document to update
            document: New document content and metadata
            collection_name: Name of collection containing document
        """
        if collection_name not in self.vector_stores:
            raise ValueError(f"Collection {collection_name} does not exist")

        # Delete existing document
        await self.delete(doc_id, collection_name)
        
        # Add updated document
        langchain_doc = self._convert_to_langchain_document(document)
        self.vector_stores[collection_name].add_documents(
            documents=[langchain_doc],
            ids=[doc_id]
        )

    async def delete(
        self, doc_ids: Union[str, List[str]], collection_name: str = "research"
    ) -> None:
        """
        Delete documents from Pinecone

        Args:
            doc_ids: Single ID or list of IDs to delete
            collection_name: Name of collection containing documents
        """
        if collection_name not in self.vector_stores:
            raise ValueError(f"Collection {collection_name} does not exist")

        if isinstance(doc_ids, str):
            doc_ids = [doc_ids]

        index = self.vector_stores[collection_name].index
        index.delete(ids=doc_ids)

    async def get_by_metadata(
        self,
        metadata_filters: Dict,
        collection_name: str = "research",
        n_results: int = 5,
    ) -> List[EnhancedDocument]:
        """
        Get documents based on metadata filters from Pinecone

        Args:
            metadata_filters: Metadata criteria to filter by
            collection_name: Name of collection to search
            n_results: Maximum number of results to return

        Returns:
            List of matching documents
        """
        if collection_name not in self.vector_stores:
            raise ValueError(f"Collection {collection_name} does not exist")

        # Pinecone supports metadata filtering natively
        results = self.vector_stores[collection_name].similarity_search(
            query="",  # Empty query for metadata-only search
            k=n_results,
            filter=metadata_filters
        )

        return [self._convert_from_langchain_document(doc) for doc in results]

    def get_collection_stats(self, collection_name: str) -> Dict[str, Any]:
        """
        Get statistics about a Pinecone collection

        Args:
            collection_name: Name of collection

        Returns:
            Dictionary containing collection statistics
        """
        if collection_name not in self.vector_stores:
            raise ValueError(f"Collection {collection_name} does not exist")

        index = self.vector_stores[collection_name].index
        stats = index.describe_index_stats()

        return {
            "total_documents": stats.total_vector_count,
            "metadata": self.collection_metadata[collection_name],
            "dimension": stats.dimension,
            "index_fullness": stats.index_fullness,
        }
    




# Example usage

# Initialize vector store
# vector_store = VectorStore()

# # Create a document
# doc = EnhancedDocument(
#     content="Your document content",
#     metadata=DocumentMetadata(
#         source="example",
#         timestamp=datetime.now(),
#         doc_type="research"
#     )
# )

# # Store document
# doc_ids = await vector_store.store(doc)

# # Search documents
# results = await vector_store.search("your query")

# No local installation required (avoiding the C++ build tools issue)
# Better scalability (hosted solution)
# Native metadata filtering
# Real-time updates without rebuild
# Built-in persistence
# Better performance for large datasets