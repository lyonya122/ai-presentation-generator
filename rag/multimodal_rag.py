import chromadb
from sentence_transformers import SentenceTransformer
import torch
from PIL import Image
import torchvision.transforms as transforms
from typing import List, Dict
import os
from config.settings import RAGConfig, KNOWLEDGE_BASE_DIR
import hashlib

class MultimodalRAG:
    """Мультимодальный RAG для поиска изображений и визуалов"""
    
    def __init__(self):
        self.client = chromadb.Client(Settings(
            persist_directory=str(KNOWLEDGE_BASE_DIR / "multimodal_db"),
            anonymized_telemetry=False
        ))
        
        # Создаем или получаем коллекцию для изображений
        self.collection = self._get_or_create_collection()
        
        # Текстовый эмбеддер
        self.text_embedder = SentenceTransformer('clip-ViT-B-32-multilingual-v1')
        
        # Трансформы для изображений
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def _get_or_create_collection(self):
        try:
            return self.client.get_collection(RAGConfig.IMAGE_COLLECTION_NAME)
        except:
            return self.client.create_collection(RAGConfig.IMAGE_COLLECTION_NAME)
    
    def add_image(self, image_path: str, description: str, metadata: Dict = None):
        """
        Добавляет изображение в базу
        
        Args:
            image_path: Путь к изображению
            description: Текстовое описание
            metadata: Метаданные
        """
        # Загружаем изображение
        image = Image.open(image_path).convert('RGB')
        
        # Создаем эмбеддинг из описания
        embedding = self.text_embedder.encode(description).tolist()
        
        # Создаем ID
        image_id = hashlib.md5(f"{image_path}{description}".encode()).hexdigest()[:16]
        
        # Добавляем в коллекцию
        self.collection.add(
            documents=[description],
            embeddings=[embedding],
            metadatas=[{
                "path": image_path,
                "type": metadata.get("type", "image"),
                **metadata
            }],
            ids=[image_id]
        )
    
    def search_images(self, query: str, k: int = 3) -> List[Dict]:
        """
        Ищет релевантные изображения по текстовому запросу
        
        Args:
            query: Текстовый запрос
            k: Количество результатов
            
        Returns:
            List[Dict]: Список найденных изображений с метаданными
        """
        if self.collection.count() == 0:
            return []
        
        # Создаем эмбеддинг запроса
        query_embedding = self.text_embedder.encode(query).tolist()
        
        # Ищем
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(k, self.collection.count())
        )
        
        # Форматируем результаты
        images = []
        if results['metadatas']:
            for i, metadata in enumerate(results['metadatas'][0]):
                images.append({
                    "path": metadata.get("path", ""),
                    "type": metadata.get("type", "image"),
                    "description": results['documents'][0][i] if results['documents'] else "",
                    "score": 1.0  # Можно добавить реальную оценку
                })
        
        return images
    
    def add_visual_library(self, visual_dir: str):
        """
        Добавляет целую библиотеку визуалов
        
        Args:
            visual_dir: Директория с визуалами
        """
        for root, dirs, files in os.walk(visual_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.svg')):
                    path = os.path.join(root, file)
                    # Используем имя файла как базовое описание
                    description = file.replace('_', ' ').replace('-', ' ').split('.')[0]
                    
                    metadata = {
                        "type": "icon" if "icon" in path.lower() else "image",
                        "filename": file,
                        "folder": os.path.basename(root)
                    }
                    
                    self.add_image(path, description, metadata)