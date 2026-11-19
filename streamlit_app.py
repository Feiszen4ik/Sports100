"""Streamlit-интерфейс для FastAPI-классификатора спорта.

Запуск (FastAPI должен уже работать):  streamlit run streamlit_app.py
"""
import os

import requests
import streamlit as st

st.set_page_config(page_title="Sports 100 Classifier", page_icon="🏆")
st.title("🏆 Sports 100 Class Classifier")
st.write("Загрузите фото, и модель (PyTorch + FastAPI) определит вид спорта из 100 классов.")

api_url = st.sidebar.text_input("Адрес FastAPI", os.getenv("API_URL", "http://127.0.0.1:8000")).rstrip("/")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", width=400)
    with st.spinner("Classifying..."):
        try:
            response = requests.post(
                f"{api_url}/predict",
                files={"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type or "image/jpeg")},
                timeout=60,
            )
        except requests.RequestException as e:
            st.error(f"Не удалось подключиться к FastAPI ({api_url}). Запущен ли сервер? Детали: {e}")
            st.stop()

    if response.status_code == 200:
        result = response.json()
        st.success(
            f"Predicted Sport: **{result['class_name']}** "
            f"(Class ID: {result['class_id']}, confidence: {result['confidence']:.1%})"
        )
        st.subheader("Top-5")
        for item in result.get("top5", []):
            st.write(f"{item['class_name']}: {item['confidence']:.1%}")
            st.progress(min(max(item["confidence"], 0.0), 1.0))
    else:
        try:
            detail = response.json().get("detail", response.text)
        except ValueError:
            detail = response.text
        st.error(f"Ошибка FastAPI ({response.status_code}): {detail}")
