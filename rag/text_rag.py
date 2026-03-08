# rag/text_rag.py
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from typing import List, Dict
import os
import hashlib
import streamlit as st

class TextRAG:
    """Текстовый RAG с ChromaDB для семантического поиска"""
    
    def __init__(self, persist_directory="./chroma_db"):
        self.persist_directory = persist_directory
        
        # Создаем директорию если её нет
        os.makedirs(persist_directory, exist_ok=True)
        
        # Инициализируем ChromaDB клиент
        self.client = chromadb.Client(Settings(
            persist_directory=persist_directory,
            anonymized_telemetry=False,
            allow_reset=True
        ))
        
        # Создаем или получаем коллекцию
        self.collection_name = "presentation_knowledge"
        self.collection = self._get_or_create_collection()
        
        # Загружаем модель для эмбеддингов
        try:
            self.embedder = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        except Exception as e:
            st.error(f"Ошибка загрузки эмбеддера: {e}")
            raise e
    
    def _get_or_create_collection(self):
        """Получает или создает коллекцию в ChromaDB"""
        try:
            return self.client.get_collection(self.collection_name)
        except:
            return self.client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
    
    def add_documents(self, documents: List[str], metadatas: List[Dict] = None):
        """
        Добавляет документы в базу знаний с эмбеддингами
        """
        if not documents:
            return
        
        # Создаем эмбеддинги
        embeddings = self.embedder.encode(documents).tolist()
        
        # Создаем ID для документов
        ids = []
        for doc in documents:
            doc_hash = hashlib.md5(doc.encode()).hexdigest()[:16]
            ids.append(doc_hash)
        
        # Добавляем в коллекцию
        self.collection.add(
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas if metadatas else [{}] * len(documents),
            ids=ids
        )
    
    def search(self, query: str, k: int = 5) -> str:
        """
        Семантический поиск релевантных документов
        """
        if self.collection.count() == 0:
            return ""
        
        # Создаем эмбеддинг запроса
        query_embedding = self.embedder.encode(query).tolist()
        
        # Ищем семантически похожие документы
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(k, self.collection.count())
        )
        
        # Объединяем результаты
        if results['documents'] and results['documents'][0]:
            return "\n\n".join(results['documents'][0])
        return ""
    
    def count(self) -> int:
        """Возвращает количество документов в базе"""
        return self.collection.count()
    
    def clear(self):
        """Очищает базу знаний"""
        try:
            self.client.delete_collection(self.collection_name)
            self.collection = self._get_or_create_collection()
        except:
            pass
