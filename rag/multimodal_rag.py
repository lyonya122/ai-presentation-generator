# rag/multimodal_rag.py
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from typing import List, Dict
import os
import hashlib
from PIL import Image
import streamlit as st

class MultimodalRAG:
    """Мультимодальный RAG для поиска изображений по смыслу"""
    
    def __init__(self, persist_directory="./multimodal_db"):
        self.persist_directory = persist_directory
        
        # Создаем директорию
        os.makedirs(persist_directory, exist_ok=True)
        
        # Инициализируем ChromaDB клиент
        self.client = chromadb.Client(Settings(
            persist_directory=persist_directory,
            anonymized_telemetry=False,
            allow_reset=True
        ))
        
        # Создаем коллекцию для изображений
        self.collection_name = "visual_knowledge"
        self.collection = self._get_or_create_collection()
        
        # Используем CLIP для мультимодальных эмбеддингов
        try:
            self.embedder = SentenceTransformer('clip-ViT-B-32-multilingual-v1')
        except Exception as e:
            st.error(f"Ошибка загрузки CLIP модели: {e}")
            raise e
    
    def _get_or_create_collection(self):
        """Получает или создает коллекцию"""
        try:
            return self.client.get_collection(self.collection_name)
        except:
            return self.client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
    
    def add_image(self, image_path: str, description: str, metadata: Dict = None):
        """
        Добавляет изображение в базу с текстовым описанием
        """
        # Создаем эмбеддинг из описания
        embedding = self.embedder.encode(description).tolist()
        
        # Создаем ID
        image_id = hashlib.md5(f"{image_path}{description}".encode()).hexdigest()[:16]
        
        # Метаданные
        meta = {
            "path": image_path,
            "type": metadata.get("type", "image") if metadata else "image",
            "description": description[:100]
        }
        if metadata:
            meta.update(metadata)
        
        # Добавляем в коллекцию
        self.collection.add(
            documents=[description],
            embeddings=[embedding],
            metadatas=[meta],
            ids=[image_id]
        )
    
    def search_images(self, query: str, k: int = 3) -> List[Dict]:
        """
        Ищет семантически релевантные изображения
        """
        if self.collection.count() == 0:
            return []
        
        # Создаем эмбеддинг запроса
        query_embedding = self.embedder.encode(query).tolist()
        
        # Ищем в базе
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(k, self.collection.count())
        )
        
        # Форматируем результаты
        images = []
        if results['metadatas'] and results['metadatas'][0]:
            for i, metadata in enumerate(results['metadatas'][0]):
                images.append({
                    "path": metadata.get("path", ""),
                    "type": metadata.get("type", "image"),
                    "description": results['documents'][0][i] if results['documents'] else "",
                    "score": 1.0  # Можно добавить реальную оценку
                })
        
        return images
    
    def count(self) -> int:
        """Возвращает количество изображений в базе"""
        return self.collection.count()
