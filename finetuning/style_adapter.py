# finetuning/style_adapter.py

class StyleAdapter:
    """Адаптер для корпоративного стиля (заглушка)"""
    
    def __init__(self):
        self.style = None
    
    def apply_style(self, prompt, style_guide):
        """
        Применяет стиль к промпту
        """
        if style_guide:
            return f"{prompt}\n\nСледуй этому стилю: {style_guide}"
        return prompt
