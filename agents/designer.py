import json
from typing import Dict, List
from agents.base_agent import BaseAgent
from config.prompts import DESIGNER_PROMPT
from rag.multimodal_rag import MultimodalRAG
from PIL import Image
import base64
from io import BytesIO

class DesignerAgent(BaseAgent):
    """Агент для дизайна слайдов с мультимодальным RAG"""
    
    def __init__(self):
        super().__init__()
        self.multimodal_rag = MultimodalRAG()
        
        # Доступные макеты слайдов
        self.layouts = {
            "title_slide": {"type": "title", "description": "Титульный слайд"},
            "content_with_image": {"type": "content", "description": "Текст слева, изображение справа"},
            "content_only": {"type": "content", "description": "Только текст"},
            "comparison": {"type": "comparison", "description": "Сравнение двух элементов"},
            "timeline": {"type": "timeline", "description": "Временная шкала"},
            "quote": {"type": "quote", "description": "Цитата или выделенная мысль"}
        }
    
    def process(self, slide_content: Dict, topic: str, 
                corporate_style: Dict = None) -> Dict:
        """
        Подбирает дизайн для слайда
        
        Args:
            slide_content: Контент слайда (из copywriter)
            topic: Общая тема
            corporate_style: Корпоративный стиль
            
        Returns:
            Dict: Дизайн-решения
        """
        # Создаем запрос для поиска визуалов
        visual_query = f"{slide_content['final_title']} {' '.join(slide_content['bullets'])}"
        
        # Ищем релевантные изображения в мультимодальной базе
        relevant_visuals = self.multimodal_rag.search_images(visual_query, k=3)
        
        # Определяем подходящий макет
        layout = self._select_layout(slide_content)
        
        # Формируем промпт для дизайнера
        visual_context = self._format_visual_context(relevant_visuals)
        
        prompt = DESIGNER_PROMPT.format(
            title=slide_content['final_title'],
            text='\n'.join(slide_content['bullets']),
            objective=slide_content.get('conclusion', ''),
            visual_context=visual_context,
            layouts=str(self.layouts)
        )
        
        # Генерируем дизайн-решения
        response = self._generate(prompt, max_new_tokens=500, temperature=0.6)
        
        # Парсим ответ
        design = self._parse_design_response(response, relevant_visuals, layout)
        
        return design
    
    def _select_layout(self, slide_content: Dict) -> str:
        """Выбирает подходящий макет на основе контента"""
        text_length = len(' '.join(slide_content['bullets']))
        
        if text_length < 50:
            return "quote"
        elif len(slide_content['bullets']) > 5:
            return "content_only"
        else:
            return "content_with_image"
    
    def _format_visual_context(self, visuals: List[Dict]) -> str:
        """Форматирует информацию о найденных визуалах"""
        context = "Доступные визуальные элементы:\n"
        for i, visual in enumerate(visuals, 1):
            context += f"{i}. {visual['description']} (тип: {visual['type']})\n"
        return context
    
    def _parse_design_response(self, response: str, 
                                visuals: List[Dict], layout: str) -> Dict:
        """Парсит ответ дизайнера"""
        try:
            design = json.loads(re.search(r'\{.*\}', response, re.DOTALL).group())
            return design
        except:
            # Возвращаем базовый дизайн если не удалось распарсить
            return {
                "layout_id": layout,
                "selected_visuals": [
                    {"type": v['type'], "source": v['path'], "placement": "right"}
                    for v in visuals[:1] if v
                ] if visuals else [],
                "color_scheme": {"primary": "#2E4053", "secondary": "#F8F9F9"},
                "notes": "Автоматически подобранный дизайн"
            }