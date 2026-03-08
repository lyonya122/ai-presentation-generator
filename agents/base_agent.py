from abc import ABC, abstractmethod
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from config.settings import ModelConfig
import streamlit as st

class BaseAgent(ABC):
    """Базовый класс для всех агентов"""
    
    def __init__(self, model_name=ModelConfig.TEXT_MODEL_NAME):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    @st.cache_resource
    def _load_model(_self):
        """Загрузка модели с квантизацией"""
        try:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=ModelConfig.LOAD_IN_4BIT,
                bnb_4bit_quant_type=ModelConfig.BNB_4BIT_QUANT_TYPE,
                bnb_4bit_compute_dtype=torch.float16
            )
            
            tokenizer = AutoTokenizer.from_pretrained(_self.model_name)
            tokenizer.pad_token = tokenizer.eos_token
            
            model = AutoModelForCausalLM.from_pretrained(
                _self.model_name,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True
            )
            
            return tokenizer, model
        except Exception as e:
            st.error(f"Ошибка загрузки модели: {e}")
            return None, None
    
    def _ensure_model_loaded(self):
        if self.tokenizer is None or self.model is None:
            self.tokenizer, self.model = self._load_model()
            return self.tokenizer is not None and self.model is not None
        return True
    
    def _generate(self, prompt, max_new_tokens=500, temperature=0.7):
        """Базовый метод генерации текста"""
        if not self._ensure_model_loaded():
            return None
            
        messages = [{"role": "user", "content": prompt}]
        formatted = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        inputs = self.tokenizer(formatted, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                top_p=0.95
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Извлекаем ответ модели
        if "[/INST]" in response:
            response = response.split("[/INST]")[-1].strip()
        
        return response
    
    @abstractmethod
    def process(self, *args, **kwargs):
        """Абстрактный метод, который должны реализовать все агенты"""
        pass