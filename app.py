import streamlit as st
import os
import tempfile
from agents.coordinator import PresentationCoordinator
from utils.document_processor import process_uploaded_files
from utils.pptx_generator import PowerPointGenerator
from rag.text_rag import TextRAG
from rag.multimodal_rag import MultimodalRAG
from config.settings import KNOWLEDGE_BASE_DIR

# Настройка страницы
st.set_page_config(
    page_title="AI Presentation Generator",
    page_icon="🎯",
    layout="wide"
)

# Заголовок
st.title("🎯 AI Presentation Generator")
st.markdown("### Мульти-агентная система создания презентаций с RAG и LangGraph")
st.markdown("---")

# Инициализация сессии
if "coordinator" not in st.session_state:
    st.session_state.coordinator = PresentationCoordinator()
if "text_rag" not in st.session_state:
    st.session_state.text_rag = TextRAG()
if "multimodal_rag" not in st.session_state:
    st.session_state.multimodal_rag = MultimodalRAG()

# Боковая панель
with st.sidebar:
    st.header("📁 Загрузка данных")
    
    # Загрузка документов
    uploaded_files = st.file_uploader(
        "Загрузите документы (txt, pdf, docx)",
        type=['txt', 'pdf', 'docx'],
        accept_multiple_files=True,
        key="document_uploader"
    )
    
    # Загрузка изображений для базы визуалов
    uploaded_images = st.file_uploader(
        "Загрузите изображения для базы визуалов",
        type=['png', 'jpg', 'jpeg', 'svg'],
        accept_multiple_files=True,
        key="image_uploader"
    )
    
    st.markdown("---")
    st.header("⚙️ Настройки")
    
    num_slides = st.slider("Количество слайдов", 3, 20, 8)
    audience = st.selectbox(
        "Целевая аудитория",
        ["general", "students", "business", "investors", "technical"]
    )
    
    use_style_adapter = st.checkbox("Использовать корпоративный стиль", False)
    
    if use_style_adapter:
        style_guide = st.text_area("Опишите корпоративный стиль", height=100)
    else:
        style_guide = ""
    
    st.markdown("---")
    st.info(
        "**Модели:**\n"
        "- Mistral-7B-Instruct (текст)\n"
        "- CLIP (мультимодальный поиск)\n"
        "- LangGraph (координация агентов)"
    )

# Основная область
col1, col2 = st.columns([2, 1])

with col1:
    topic = st.text_input(
        "📝 Введите тему презентации:",
        placeholder="Например: Будущее искусственного интеллекта в образовании"
    )
    
    # Кнопка генерации
    if st.button("🚀 Создать презентацию", type="primary", use_container_width=True):
        if not topic:
            st.error("Введите тему презентации!")
        else:
            # Создаем placeholder для статуса
            status_container = st.container()
            
            with status_container:
                # Обработка загруженных файлов
                if uploaded_files:
                    with st.status("📄 Обработка документов..."):
                        documents_text = process_uploaded_files(uploaded_files)
                        # Добавляем в RAG
                        st.session_state.text_rag.add_documents(
                            [documents_text],
                            [{"source": "uploaded", "topic": topic}]
                        )
                
                # Обработка изображений
                if uploaded_images:
                    with st.status("🖼️ Обработка изображений..."):
                        for img in uploaded_images:
                            # Сохраняем временно
                            with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp:
                                tmp.write(img.getvalue())
                                st.session_state.multimodal_rag.add_image(
                                    tmp.name,
                                    f"Image from user: {img.name}",
                                    {"source": "uploaded"}
                                )
                
                # Создание презентации
                with st.status("🤖 Агенты работают..."):
                    presentation_data = st.session_state.coordinator.create_presentation(
                        topic=topic,
                        num_slides=num_slides,
                        audience=audience,
                        context=documents_text if uploaded_files else ""
                    )
                
                # Генерация PowerPoint
                with st.status("📊 Генерация PowerPoint..."):
                    generator = PowerPointGenerator()
                    
                    # Сохраняем во временный файл
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.pptx') as tmp:
                        output_path = generator.create_presentation(
                            presentation_data,
                            tmp.name
                        )
                
                st.success("✅ Презентация готова!")
                
                # Кнопка скачивания
                with open(output_path, 'rb') as file:
                    st.download_button(
                        label="📥 Скачать презентацию",
                        data=file,
                        file_name=f"{topic.replace(' ', '_')}.pptx",
                        mime="application/vnd.openxmlformats-officedocument.presentationml.presentation",
                        use_container_width=True
                    )
                
                # Показываем метрики
                col_q1, col_q2, col_q3 = st.columns(3)
                with col_q1:
                    st.metric("Качество", f"{presentation_data['quality_score']:.1f}/10")
                with col_q2:
                    st.metric("Слайдов", presentation_data['num_slides'])
                with col_q3:
                    st.metric("Итераций", presentation_data['metadata']['iterations'])

with col2:
    st.info("ℹ️ Как это работает")
    st.markdown("""
    ### 🎯 Мульти-агентная архитектура
    
    **1. Агент-структуризатор**
    - Создает логический план
    - Определяет цели слайдов
    - Использует LangGraph для итераций
    
    **2. Агент-копирайтер**
    - Пишет тексты для слайдов
    - Использует RAG из документов
    - Адаптирует tone of voice
    
    **3. Агент-дизайнер**
    - Подбирает визуалы
    - Мультимодальный RAG
    - Выбирает макеты
    
    **🚀 Продвинутые функции:**
    - LangGraph координация
    - Мультимодальный RAG
    - LoRA дообучение под стиль
    - Векторный поиск изображений
    """)
    
    # Показываем примеры из базы знаний
    with st.expander("📚 База знаний"):
        st.markdown("**Текстовые документы:**")
        text_count = st.session_state.text_rag.collection.count()
        st.metric("Документов в RAG", text_count)
        
        st.markdown("**Визуальные элементы:**")
        image_count = st.session_state.multimodal_rag.collection.count()
        st.metric("Изображений в базе", image_count)

# Добавляем вкладки с детальной информацией
tab1, tab2, tab3 = st.tabs(["📋 Структура", "✍️ Контент", "🎨 Дизайн"])

if 'presentation_data' in locals():
    with tab1:
        st.json(presentation_data['structure'])
    
    with tab2:
        st.json(presentation_data['content'])
    
    with tab3:
        st.json(presentation_data['design'])