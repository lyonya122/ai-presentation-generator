import os
from pathlib import Path

# Пути к проекту
BASE_DIR = Path(__file__).parent.parent
KNOWLEDGE_BASE_DIR = BASE_DIR / "knowledge_base"
PRESENTATIONS_DIR = BASE_DIR / "presentations"

# Создаем директории если их нет
os.makedirs(KNOWLEDGE_BASE_DIR / "documents", exist_ok=True)
os.makedirs(KNOWLEDGE_BASE_DIR / "visuals", exist_ok=True)
os.makedirs(PRESENTATIONS_DIR, exist_ok=True)

# Настройки моделей
class ModelConfig:
    # Основная текстовая модель
    TEXT_MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"
    
    # Мультимодальная модель для понимания изображений
    VLM_MODEL_NAME = "llava-hf/llava-1.5-7b-hf"
    
    # Модель для эмбеддингов
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    
    # Настройки квантизации (экономия памяти)
    LOAD_IN_4BIT = True
    BNB_4BIT_QUANT_TYPE = "nf4"
    BNB_4BIT_COMPUTE_DTYPE = "float16"

# Настройки RAG
class RAGConfig:
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    TOP_K_RESULTS = 5
    COLLECTION_NAME = "presentation_knowledge"
    
    # Для мультимодального RAG
    IMAGE_COLLECTION_NAME = "visual_knowledge"
    IMAGE_EMBEDDING_SIZE = 512

# Настройки LoRA дообучения
class LoRAConfig:
    R = 8  # ранг LoRA
    ALPHA = 16  # scaling factor
    DROPOUT = 0.1
    TARGET_MODULES = ["q_proj", "v_proj"]  # целевые модули для LoRA
    
# Настройки LangGraph
class LangGraphConfig:
    MAX_ITERATIONS = 3  # макс. количество итераций агентов
    TEMPERATURE = 0.7  # температура для генерации