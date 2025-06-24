import streamlit as st
import joblib
import pdfplumber
import re
import os
from openai import OpenAI

api_key = os.environ.get("OPENAI_API_KEY")

if not api_key:
    raise ValueError("OPENAI_API_KEY not found in environment variables.")

client = OpenAI(api_key=api_key)

model = joblib.load('random_forest_resume_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

def extract_text_from_pdf(file):
    with pdfplumber.open(file) as pdf:
        pages = [page.extract_text() or "" for page in pdf.pages]
    return "\n\n".join(pages)

def preprocess_text(text):
    text = text.lower().strip()
    text = re.sub(r'\s+', ' ', text)
    return text

def explain_prediction_with_advice(text, score):
    prompt = (
        f"This resume was rated {score:.2f}/5 by an AI model.\n\n"
        f"Resume:\n{text}\n\n"
        f"Please provide a clear, concise analysis of this resume, "
        f"including constructive advice on how the candidate can improve their resume. "
        f"Focus only on the content of the resume."
    )
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": "You are an HR assistant that analyzes resumes and gives constructive feedback."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.5,
            max_tokens=800
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error when accessing GPT: {e}"

st.title("Personal HR Assistant")

uploaded_file = st.file_uploader("Upload one PDF resume", type=['pdf'], accept_multiple_files=False)

if uploaded_file:
    raw_text = extract_text_from_pdf(uploaded_file)
    cleaned_text = preprocess_text(raw_text)

    features = vectorizer.transform([cleaned_text])
    score = model.predict(features)[0]

    st.success(f"Overall resume rating: **{score:.2f}** / 5")

    with st.spinner("Analyzing resume and generating recommendations..."):
        analysis = explain_prediction_with_advice(raw_text, score)

    st.markdown("#### Resume Analysis and Improvement Tips:")
    st.info(analysis)
