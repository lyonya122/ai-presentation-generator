import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset
from config.settings import LoRAConfig
import json

class LoRATrainer:
    """Класс для дообучения модели через LoRA"""
    
    def __init__(self, base_model_name: str):
        self.base_model_name = base_model_name
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        self.model = None
        self.lora_config = None
    
    def prepare_model_for_lora(self):
        """Подготавливает модель для LoRA дообучения"""
        # Загружаем базовую модель
        self.model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        # Конфигурация LoRA
        self.lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=LoRAConfig.R,
            lora_alpha=LoRAConfig.ALPHA,
            lora_dropout=LoRAConfig.DROPOUT,
            target_modules=LoRAConfig.TARGET_MODULES
        )
        
        # Применяем LoRA
        self.model = get_peft_model(self.model, self.lora_config)
        self.model.print_trainable_parameters()
        
        return self.model
    
    def prepare_dataset(self, presentation_examples: List[Dict]) -> Dataset:
        """
        Подготавливает датасет из примеров презентаций
        
        Args:
            presentation_examples: Список примеров презентаций в формате:
                [{
                    "topic": "тема",
                    "structure": [...],
                    "slides": [...],
                    "style": "корпоративный стиль"
                }]
        """
        texts = []
        for example in presentation_examples:
            # Форматируем пример в промпт-ответ
            prompt = f"Тема: {example['topic']}\nСоздай презентацию в стиле {example['style']}"
            response = json.dumps({
                "structure": example['structure'],
                "slides": example['slides']
            }, ensure_ascii=False)
            
            texts.append(f"<s>[INST] {prompt} [/INST] {response}</s>")
        
        # Токенизируем
        encodings = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=2048,
            return_tensors="pt"
        )
        
        return Dataset.from_dict({
            "input_ids": encodings["input_ids"],
            "attention_mask": encodings["attention_mask"],
            "labels": encodings["input_ids"]
        })
    
    def train(self, train_dataset: Dataset, output_dir: str):
        """
        Запускает дообучение
        
        Args:
            train_dataset: Датасет для обучения
            output_dir: Директория для сохранения модели
        """
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=3,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            warmup_steps=100,
            logging_steps=10,
            learning_rate=2e-4,
            fp16=True,
            save_strategy="epoch"
        )
        
        # Создаем тренера
        from transformers import Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset
        )
        
        # Обучаем
        trainer.train()
        
        # Сохраняем адаптер
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        return output_dir