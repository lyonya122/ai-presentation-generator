import json
import re
from typing import Dict, List
from agents.base_agent import BaseAgent
from config.prompts import COPYWRITER_PROMPT
from rag.text_rag import TextRAG
from finetuning.style_adapter import StyleAdapter

class CopywriterAgent(BaseAgent):
    """Агент для генерации текстов слайдов"""
    
    def __init__(self, use_style_adapter: bool = False):
        super().__init__()
        self.rag = TextRAG()
        self.style_adapter = StyleAdapter() if use_style_adapter else None
    
    def process(self, slide_info: Dict, topic: str, audience: str = "general",
                style_guide: str = "") -> Dict:
        """
        Генерирует текст для слайда
        
        Args:
            slide_info: Информация о слайде из структуры
            topic: Общая тема
            audience: Целевая аудитория
            style_guide: Гайд по стилю (для дообучения)
            
        Returns:
            Dict: Сгенерированный контент
        """
        # Получаем релевантный контекст для этого слайда
        search_query = f"{slide_info['title']} {' '.join(slide_info['key_points'])}"
        context = self.rag.search(search_query, k=3)
        
        # Применяем стиль если есть
        if self.style_adapter and style_guide:
            styled_prompt = self.style_adapter.apply_style(
                COPYWRITER_PROMPT, style_guide
            )
        else:
            styled_prompt = COPYWRITER_PROMPT
        
        # Формируем промпт
        prompt = styled_prompt.format(
            title=slide_info['title'],
            objective=slide_info['objective'],
            key_points=slide_info['key_points'],
            topic=topic,
            audience=audience,
            context=context,
            style_guide=style_guide
        )
        
        # Генерируем текст
        response = self._generate(prompt, max_new_tokens=500, temperature=0.7)
        
        # Парсим результат
        content = self._parse_content(response, slide_info)
        
        return content
    
    def _parse_content(self, response: str, slide_info: Dict) -> Dict:
        """Парсит сгенерированный контент"""
        try:
            # Пробуем распарсить JSON
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                content = json.loads(json_match.group())
                return content
        except:
            pass
        
        # Если JSON не найден, создаем структуру из текста
        lines = response.strip().split('\n')
        bullets = []
        conclusion = ""
        
        for line in lines:
            line = line.strip()
            if line.startswith(('- ', '• ', '* ')):
                bullets.append(line[2:].strip())
            elif 'conclusion' in line.lower() or 'вывод' in line.lower():
                conclusion = line.split(':')[-1].strip() if ':' in line else line
        
        return {
            "final_title": slide_info['title'],
            "bullets": bullets if bullets else slide_info['key_points'],
            "speaker_notes": "",
            "conclusion": conclusion or "Ключевой вывод по слайду"
        }