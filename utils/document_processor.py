# utils/document_processor.py
import streamlit as st

def process_uploaded_files(uploaded_files):
    """
    Обработка загруженных файлов
    """
    if not uploaded_files:
        return ""
    
    combined_text = ""
    for file in uploaded_files:
        try:
            # Для TXT файлов
            if file.name.endswith('.txt'):
                stringio = file.getvalue().decode("utf-8")
                combined_text += f"\n\n--- Из файла {file.name} ---\n\n"
                combined_text += stringio
            
            # Для PDF (заглушка)
            elif file.name.endswith('.pdf'):
                combined_text += f"\n\n--- PDF файл {file.name} (обработка в разработке) ---\n\n"
            
            # Для DOCX (заглушка)
            elif file.name.endswith('.docx'):
                combined_text += f"\n\n--- DOCX файл {file.name} (обработка в разработке) ---\n\n"
                
        except Exception as e:
            st.error(f"Ошибка при обработке файла {file.name}: {str(e)}")
    
    return combined_text

def truncate_text(text, max_length=2000):
    """
    Обрезает текст до максимальной длины
    """
    if len(text) <= max_length:
        return text
    
    return text[:max_length] + "..."
