"""
MongoDB Database Persistence Layer for Learn RAG.
Stores document metadata, chunk records, and agent thinking/reasoning sessions.
Provides automatic fallback to an in-memory store if MongoDB is offline.
"""

import datetime
import logging
from typing import Dict, Any, List, Optional, Tuple
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError

logger = logging.getLogger(__name__)


class InMemoryStore:
    """Fallback in-memory storage when MongoDB service is unreachable."""

    def __init__(self):
        self.documents: Dict[str, Dict[str, Any]] = {}
        self.chunks: Dict[str, List[Dict[str, Any]]] = {}
        self.thinking_sessions: List[Dict[str, Any]] = []
        self.chat_sessions: Dict[str, Dict[str, Any]] = {}

    def save_document(self, doc_data: Dict[str, Any]) -> None:
        self.documents[doc_data["doc_id"]] = doc_data

    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        return self.documents.get(doc_id)

    def list_documents(self) -> List[Dict[str, Any]]:
        return list(self.documents.values())

    def delete_document(self, doc_id: str) -> None:
        self.documents.pop(doc_id, None)
        self.chunks.pop(doc_id, None)

    def save_chunks(self, doc_id: str, chunk_records: List[Dict[str, Any]]) -> int:
        self.chunks[doc_id] = chunk_records
        return len(chunk_records)

    def get_chunks(self, doc_id: str) -> List[Dict[str, Any]]:
        return self.chunks.get(doc_id, [])

    def save_thinking_session(self, session_data: Dict[str, Any]) -> None:
        self.thinking_sessions.append(session_data)

    def get_recent_sessions(self, limit: int = 20) -> List[Dict[str, Any]]:
        return self.thinking_sessions[-limit:]

    def save_chat_session(self, session_data: Dict[str, Any]) -> None:
        self.chat_sessions[session_data["session_id"]] = session_data

    def get_chat_sessions(self, limit: int = 50) -> List[Dict[str, Any]]:
        sessions = sorted(self.chat_sessions.values(), key=lambda s: s.get("updated_at", ""), reverse=True)
        return sessions[:limit]

    def delete_chat_session(self, session_id: str) -> None:
        self.chat_sessions.pop(session_id, None)


class MongoRAGDatabase:
    """
    Manages MongoDB persistence for documents, chunks, and thinking reasoning traces.
    Safely degrades to InMemoryStore if MongoDB is offline.
    """

    def __init__(self, uri: str = "mongodb://localhost:27017", db_name: str = "papermind_rag"):
        self.uri = uri
        self.db_name = db_name
        self.client: Optional[MongoClient] = None
        self.db = None
        self._is_connected = False
        self._connection_message = "Not connected"
        self._in_memory = InMemoryStore()
        
        self.connect()

    def connect(self) -> Tuple[bool, str]:
        """Attempts to connect and ping the MongoDB cluster."""
        try:
            self.client = MongoClient(self.uri, serverSelectionTimeoutMS=2000)
            # Test ping
            self.client.admin.command("ping")
            self.db = self.client[self.db_name]
            self._is_connected = True
            self._connection_message = f"Connected to MongoDB ({self.db_name})"
            
            # Setup indexes
            self.db.documents.create_index("doc_id", unique=True)
            self.db.chunks.create_index([("doc_id", 1), ("chunk_index", 1)])
            self.db.thinking_sessions.create_index("created_at")
            
            logger.info(f"MongoDB connection established: {self._connection_message}")
            return True, self._connection_message
        except (ConnectionFailure, ServerSelectionTimeoutError, Exception) as exc:
            self._is_connected = False
            self._connection_message = f"MongoDB offline or unreachable: {exc}. Using in-memory fallback."
            logger.warning(self._connection_message)
            return False, self._connection_message

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    @property
    def connection_status(self) -> Dict[str, Any]:
        return {
            "connected": self._is_connected,
            "message": self._connection_message,
            "db_name": self.db_name,
            "storage_mode": "MongoDB" if self._is_connected else "In-Memory Fallback"
        }

    def save_document(
        self,
        doc_id: str,
        filename: str,
        file_size_bytes: int,
        total_pages: int,
        total_chunks: int,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Saves or updates document metadata."""
        record = {
            "doc_id": doc_id,
            "filename": filename,
            "file_size_bytes": file_size_bytes,
            "total_pages": total_pages,
            "total_chunks": total_chunks,
            "metadata": metadata or {},
            "updated_at": datetime.datetime.now(datetime.timezone.utc).isoformat()
        }
        
        if self._is_connected and self.db is not None:
            try:
                self.db.documents.update_one(
                    {"doc_id": doc_id},
                    {"$set": record, "$setOnInsert": {"created_at": datetime.datetime.now(datetime.timezone.utc).isoformat()}},
                    upsert=True
                )
            except Exception as exc:
                logger.error(f"Error saving document to MongoDB: {exc}")
                self._in_memory.save_document(record)
        else:
            self._in_memory.save_document(record)
            
        return record

    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves document record by doc_id."""
        if self._is_connected and self.db is not None:
            try:
                doc = self.db.documents.find_one({"doc_id": doc_id}, {"_id": 0})
                if doc:
                    return doc
            except Exception as exc:
                logger.error(f"Error reading document from MongoDB: {exc}")
        return self._in_memory.get_document(doc_id)

    def list_documents(self) -> List[Dict[str, Any]]:
        """Lists all registered documents."""
        if self._is_connected and self.db is not None:
            try:
                return list(self.db.documents.find({}, {"_id": 0}))
            except Exception as exc:
                logger.error(f"Error listing documents from MongoDB: {exc}")
        return self._in_memory.list_documents()

    def delete_document(self, doc_id: str) -> bool:
        """Deletes a document and its chunks."""
        self._in_memory.delete_document(doc_id)
        if self._is_connected and self.db is not None:
            try:
                self.db.documents.delete_one({"doc_id": doc_id})
                self.db.chunks.delete_many({"doc_id": doc_id})
                return True
            except Exception as exc:
                logger.error(f"Error deleting document from MongoDB: {exc}")
        return True

    def save_chunks(self, doc_id: str, chunks: List[Any]) -> int:
        """
        Persists chunks into MongoDB.
        Accepts list of LangChain Document objects or raw dicts.
        """
        records: List[Dict[str, Any]] = []
        for idx, chunk in enumerate(chunks):
            if hasattr(chunk, "page_content"):
                content = chunk.page_content
                meta = dict(chunk.metadata)
            else:
                content = chunk.get("content", chunk.get("page_content", ""))
                meta = dict(chunk.get("metadata", {}))

            chunk_id = meta.get("chunk_id", f"{doc_id}_chunk_{idx}")
            page = meta.get("page", meta.get("page_number", 1))

            records.append({
                "doc_id": doc_id,
                "chunk_id": chunk_id,
                "chunk_index": idx,
                "page": page,
                "page_number": page,
                "content": content,
                "length": len(content),
                "metadata": meta,
                "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat()
            })

        self._in_memory.save_chunks(doc_id, records)

        if self._is_connected and self.db is not None:
            try:
                # Remove existing chunks for this doc_id
                self.db.chunks.delete_many({"doc_id": doc_id})
                if records:
                    self.db.chunks.insert_many(records)
                return len(records)
            except Exception as exc:
                logger.error(f"Error saving chunks to MongoDB: {exc}")
        return len(records)

    def get_chunks(self, doc_id: str) -> List[Dict[str, Any]]:
        """Retrieves all chunk records for a given doc_id."""
        if self._is_connected and self.db is not None:
            try:
                chunks = list(self.db.chunks.find({"doc_id": doc_id}, {"_id": 0}).sort("chunk_index", 1))
                if chunks:
                    return chunks
            except Exception as exc:
                logger.error(f"Error reading chunks from MongoDB: {exc}")
        return self._in_memory.get_chunks(doc_id)

    def save_thinking_session(
        self,
        query: str,
        thinking_trace: List[Dict[str, Any]],
        answer: str,
        citations: Dict[str, Any],
        chunks: List[Dict[str, Any]],
        doc_id: Optional[str] = None,
        model_used: str = ""
    ) -> Dict[str, Any]:
        """Records an agentic thinking & generation trace in MongoDB."""
        session_record = {
            "query": query,
            "doc_id": doc_id,
            "model_used": model_used,
            "thinking_trace": thinking_trace,
            "answer": answer,
            "citations": citations,
            "chunk_count": len(chunks),
            "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat()
        }

        self._in_memory.save_thinking_session(session_record)

        if self._is_connected and self.db is not None:
            try:
                self.db.thinking_sessions.insert_one(session_record)
                # Remove internal _id for caller convenience
                session_record.pop("_id", None)
            except Exception as exc:
                logger.error(f"Error saving thinking session to MongoDB: {exc}")

        return session_record

    def get_recent_sessions(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Returns the most recent thinking sessions."""
        if self._is_connected and self.db is not None:
            try:
                return list(self.db.thinking_sessions.find({}, {"_id": 0}).sort("created_at", -1).limit(limit))
            except Exception as exc:
                logger.error(f"Error fetching thinking sessions from MongoDB: {exc}")
        return self._in_memory.get_recent_sessions(limit)

    def save_chat_session(
        self,
        session_id: str,
        title: str,
        messages: List[Dict[str, Any]],
        doc_id: Optional[str] = None,
        filename: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Saves or updates a full chat conversation session."""
        now = datetime.datetime.now(datetime.timezone.utc).isoformat()
        record = {
            "session_id": session_id,
            "title": title,
            "doc_id": doc_id,
            "filename": filename,
            "messages": messages,
            "message_count": len(messages),
            "updated_at": now,
        }
        self._in_memory.save_chat_session(record)

        if self._is_connected and self.db is not None:
            try:
                self.db.chat_sessions.create_index("session_id", unique=True)
                self.db.chat_sessions.update_one(
                    {"session_id": session_id},
                    {
                        "$set": record,
                        "$setOnInsert": {"created_at": now},
                    },
                    upsert=True,
                )
            except Exception as exc:
                logger.error(f"Error saving chat session to MongoDB: {exc}")
        return record

    def get_chat_sessions(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Returns recent chat sessions ordered by most recently updated."""
        if self._is_connected and self.db is not None:
            try:
                return list(
                    self.db.chat_sessions.find({}, {"_id": 0}).sort("updated_at", -1).limit(limit)
                )
            except Exception as exc:
                logger.error(f"Error fetching chat sessions from MongoDB: {exc}")
        return self._in_memory.get_chat_sessions(limit)

    def delete_chat_session(self, session_id: str) -> bool:
        """Deletes a specific chat session by session_id."""
        self._in_memory.delete_chat_session(session_id)
        if self._is_connected and self.db is not None:
            try:
                self.db.chat_sessions.delete_one({"session_id": session_id})
            except Exception as exc:
                logger.error(f"Error deleting chat session from MongoDB: {exc}")
        return True


# Singleton database instance cache
_GLOBAL_DB_INSTANCE: Optional[MongoRAGDatabase] = None


def get_rag_database(uri: Optional[str] = None, db_name: Optional[str] = None) -> MongoRAGDatabase:
    """Returns a singleton instance of the MongoRAGDatabase."""
    global _GLOBAL_DB_INSTANCE
    from rag.config import get_config
    config = get_config()

    target_uri = uri or config.mongodb_uri
    target_name = db_name or config.mongodb_db_name

    if _GLOBAL_DB_INSTANCE is None or _GLOBAL_DB_INSTANCE.uri != target_uri:
        _GLOBAL_DB_INSTANCE = MongoRAGDatabase(uri=target_uri, db_name=target_name)
    return _GLOBAL_DB_INSTANCE
