from typing import Dict, List, Any
import streamlit as st
from agents.structurizer import StructurizerAgent
from agents.copywriter import CopywriterAgent
from agents.designer import DesignerAgent
from config.settings import LangGraphConfig
from config.prompts import COORDINATOR_SYSTEM_PROMPT

class PresentationCoordinator:
    """
    Координатор агентов с использованием LangGraph-подобного подхода
    """
    
    def __init__(self):
        self.structurizer = StructurizerAgent()
        self.copywriter = CopywriterAgent()
        self.designer = DesignerAgent()
        
        # Состояние процесса
        self.state = {
            "topic": "",
            "audience": "general",
            "num_slides": 5,
            "structure": [],
            "content": [],
            "design": [],
            "iteration": 0,
            "quality_score": 0,
            "feedback": []
        }
    
    def create_presentation(self, topic: str, num_slides: int, 
                           audience: str = "general",
                           context: str = "") -> Dict[str, Any]:
        """
        Основной метод создания презентации через координацию агентов
        """
        self.state["topic"] = topic
        self.state["num_slides"] = num_slides
        self.state["audience"] = audience
        
        # Шаг 1: Структуризация
        with st.status("🤔 Агент-структуризатор создает план..."):
            self.state["structure"] = self.structurizer.process(
                topic, num_slides, audience, context
            )
            self.state["iteration"] += 1
        
        # Шаг 2: Итеративное улучшение структуры
        self._refine_structure()
        
        # Шаг 3: Генерация контента для каждого слайда
        self.state["content"] = []
        for i, slide in enumerate(self.state["structure"]):
            with st.status(f"✍️ Агент-копирайтер пишет слайд {i+1}..."):
                content = self.copywriter.process(
                    slide, topic, audience
                )
                self.state["content"].append(content)
        
        # Шаг 4: Дизайн для каждого слайда
        self.state["design"] = []
        for i, (slide, content) in enumerate(zip(self.state["structure"], 
                                                  self.state["content"])):
            with st.status(f"🎨 Агент-дизайнер оформляет слайд {i+1}..."):
                slide_with_content = {**slide, **content}
                design = self.designer.process(slide_with_content, topic)
                self.state["design"].append(design)
        
        # Шаг 5: Финальная проверка качества
        self._quality_check()
        
        return self._prepare_final_output()
    
    def _refine_structure(self):
        """Итеративное улучшение структуры"""
        for iteration in range(LangGraphConfig.MAX_ITERATIONS):
            # Проверяем качество структуры
            issues = self._analyze_structure_issues()
            
            if not issues:
                break
            
            # Улучшаем структуру на основе найденных проблем
            with st.status(f"🔄 Улучшение структуры (итерация {iteration + 1})..."):
                self.state["structure"] = self.structurizer.process(
                    self.state["topic"],
                    self.state["num_slides"],
                    self.state["audience"],
                    f"Предыдущая структура: {self.state['structure']}\n"
                    f"Проблемы: {issues}"
                )
    
    def _analyze_structure_issues(self) -> List[str]:
        """Анализирует структуру на наличие проблем"""
        issues = []
        
        if len(self.state["structure"]) != self.state["num_slides"]:
            issues.append("Неверное количество слайдов")
        
        # Проверяем наличие введения и заключения
        if self.state["structure"]:
            first_slide = self.state["structure"][0]
            if "введ" not in first_slide["title"].lower():
                issues.append("Отсутствует сильное введение")
            
            last_slide = self.state["structure"][-1]
            if "заключ" not in last_slide["title"].lower():
                issues.append("Отсутствует заключение")
        
        return issues
    
    def _quality_check(self):
        """Финальная проверка качества"""
        # Оцениваем согласованность
        self.state["quality_score"] = self._calculate_quality_score()
        
        # Генерируем фидбек
        self.state["feedback"] = self._generate_feedback()
    
    def _calculate_quality_score(self) -> float:
        """Рассчитывает оценку качества"""
        score = 10.0
        
        # Проверяем длину контента
        for content in self.state["content"]:
            if len(content.get("bullets", [])) < 2:
                score -= 0.5
        
        # Проверяем наличие дизайна
        for design in self.state["design"]:
            if not design.get("selected_visuals"):
                score -= 0.3
        
        return max(0, score)
    
    def _generate_feedback(self) -> List[str]:
        """Генерирует фидбек по презентации"""
        feedback = []
        
        if self.state["quality_score"] < 7:
            feedback.append("Рекомендуется доработать некоторые слайды")
        
        if self.state["quality_score"] > 9:
            feedback.append("Презентация высокого качества!")
        
        return feedback
    
    def _prepare_final_output(self) -> Dict[str, Any]:
        """Подготавливает финальный вывод"""
        return {
            "topic": self.state["topic"],
            "audience": self.state["audience"],
            "num_slides": self.state["num_slides"],
            "structure": self.state["structure"],
            "content": self.state["content"],
            "design": self.state["design"],
            "quality_score": self.state["quality_score"],
            "feedback": self.state["feedback"],
            "metadata": {
                "iterations": self.state["iteration"],
                "agents_used": ["structurizer", "copywriter", "designer"]
            }
        }