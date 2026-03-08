import json
import re
from typing import List, Dict, Any
from agents.base_agent import BaseAgent
from config.prompts import STRUCTURIZER_PROMPT
from rag.text_rag import TextRAG

class StructurizerAgent(BaseAgent):
    """Агент для создания структуры презентации"""
    
    def __init__(self):
        super().__init__()
        self.rag = TextRAG()
    
    def process(self, topic: str, num_slides: int, audience: str = "general", 
                additional_context: str = "") -> List[Dict[str, Any]]:
        """
        Создает структуру презентации
        
        Args:
            topic: Тема презентации
            num_slides: Количество слайдов
            audience: Целевая аудитория
            additional_context: Дополнительный контекст
            
        Returns:
            List[Dict]: Структура презентации
        """
        # Получаем релевантный контекст из RAG
        rag_context = self.rag.search(topic, k=5)
        
        # Формируем промпт
        prompt = STRUCTURIZER_PROMPT.format(
            topic=topic,
            audience=audience,
            num_slides=num_slides,
            context=rag_context + "\n" + additional_context
        )
        
        # Генерируем структуру
        response = self._generate(prompt, max_new_tokens=1000, temperature=0.7)
        
        # Парсим JSON из ответа
        structure = self._parse_json_response(response)
        
        # Валидируем и дополняем структуру
        structure = self._validate_structure(structure, num_slides, topic)
        
        return structure
    
    def _parse_json_response(self, response: str) -> List[Dict]:
        """Парсит JSON из ответа модели"""
        try:
            # Ищем JSON в ответе
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                structure = json.loads(json_match.group())
                return structure
        except:
            pass
        
        # Если не удалось распарсить, создаем базовую структуру
        return self._create_fallback_structure()
    
    def _validate_structure(self, structure: List[Dict], expected_slides: int, topic: str) -> List[Dict]:
        """Валидирует и исправляет структуру"""
        if not structure or len(structure) != expected_slides:
            return self._create_fallback_structure(topic, expected_slides)
        
        # Проверяем наличие обязательных полей
        validated = []
        for i, slide in enumerate(structure):
            validated_slide = {
                "slide_number": i + 1,
                "title": slide.get("title", f"Слайд {i+1}"),
                "objective": slide.get("objective", "Донести ключевую информацию"),
                "key_points": slide.get("key_points", ["Основная идея"]),
                "suggested_visual": slide.get("suggested_visual", "Текстовый слайд"),
                "transition_to_next": slide.get("transition_to_next", "Логический переход")
            }
            validated.append(validated_slide)
        
        return validated
    
    def _create_fallback_structure(self, topic: str, num_slides: int = 5) -> List[Dict]:
        """Создает базовую структуру если генерация не удалась"""
        structure = [
            {
                "slide_number": 1,
                "title": topic,
                "objective": "Привлечь внимание аудитории и представить тему",
                "key_points": ["Проблема/вопрос", "Актуальность темы"],
                "suggested_visual": "Заглавное изображение",
                "transition_to_next": "Переход к основной части"
            }
        ]
        
        # Добавляем основные слайды
        for i in range(2, num_slides):
            structure.append({
                "slide_number": i,
                "title": f"Аспект {i-1}",
                "objective": f"Рассмотреть аспект {i-1} темы",
                "key_points": ["Ключевой пункт 1", "Ключевой пункт 2", "Ключевой пункт 3"],
                "suggested_visual": "Диаграмма или схема",
                "transition_to_next": "Переход к следующему аспекту"
            })
        
        # Добавляем заключение
        structure.append({
            "slide_number": num_slides,
            "title": "Заключение",
            "objective": "Подвести итоги и дать рекомендации",
            "key_points": ["Основные выводы", "Рекомендации", "Вопросы?"],
            "suggested_visual": "Итоговая схема",
            "transition_to_next": "Завершение презентации"
        })
        
        return structure[:num_slides]