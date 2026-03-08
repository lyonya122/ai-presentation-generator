import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from typing import List, Dict
import os
from config.settings import RAGConfig, KNOWLEDGE_BASE_DIR
import hashlib
import pickle

class TextRAG:
    """Текстовый RAG для поиска информации в документах"""
    
    def __init__(self):
        self.client = chromadb.Client(Settings(
            persist_directory=str(KNOWLEDGE_BASE_DIR / "chroma_db"),
            anonymized_telemetry=False
        ))
        
        # Создаем или получаем коллекцию
        self.collection = self._get_or_create_collection()
        
        # Загружаем модель для эмбеддингов
        self.embedder = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    
    def _get_or_create_collection(self):
        """Получает или создает коллекцию в ChromaDB"""
        try:
            return self.client.get_collection(RAGConfig.COLLECTION_NAME)
        except:
            return self.client.create_collection(RAGConfig.COLLECTION_NAME)
    
    def add_documents(self, documents: List[str], metadatas: List[Dict] = None):
        """
        Добавляет документы в базу знаний
        
        Args:
            documents: Список текстов документов
            metadatas: Метаданные для каждого документа
        """
        # Создаем эмбеддинги
        embeddings = self.embedder.encode(documents).tolist()
        
        # Создаем ID для документов
        ids = [hashlib.md5(doc.encode()).hexdigest()[:16] for doc in documents]
        
        # Добавляем в коллекцию
        self.collection.add(
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas if metadatas else [{}] * len(documents),
            ids=ids
        )
    
    def search(self, query: str, k: int = 5) -> str:
        """
        Ищет релевантные документы по запросу
        
        Args:
            query: Поисковый запрос
            k: Количество результатов
            
        Returns:
            str: Объединенные тексты найденных документов
        """
        if self.collection.count() == 0:
            return ""
        
        # Создаем эмбеддинг запроса
        query_embedding = self.embedder.encode(query).tolist()
        
        # Ищем
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(k, self.collection.count())
        )
        
        # Объединяем результаты
        if results['documents']:
            return "\n\n".join(results['documents'][0])
        return ""
    
    def clear(self):
        """Очищает базу знаний"""
        self.client.delete_collection(RAGConfig.COLLECTION_NAME)
        self.collection = self._get_or_create_collection()