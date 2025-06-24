import streamlit as st
import joblib
import pdfplumber
import re
import numpy as np
from openai import OpenAI
import os

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
def explain_prediction(text, score):
    prompt = (
        f"The following resume was rated {score:.2f}/5 by an AI model.\n\n"
        f"Resume:\n{text}\n\n"
        # f"Briefly explain what factors in the resume might have contributed to this rating. "
        # f"Be concise, clear, and use bullet points if helpful. Only focus on resume content."
        f"Based on this resume, list 5 strengths and 5 weaknesses of the candidate. "
        f"Focus only on the resume content and keep the response concise and clear."
    )

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": "You are an HR assistant that explains AI resume evaluations based on resume content."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.5,
            max_tokens=400
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error when accessing GPT: {e}"

   


st.title("HR Assistant")

uploaded_files = st.file_uploader("Download PDF resume: ", accept_multiple_files=True, type=['pdf'])

if uploaded_files:
    resumes = []
    for file in uploaded_files:
        raw_text = extract_text_from_pdf(file)
        cleaned_text = preprocess_text(raw_text)
        resumes.append((file.name, cleaned_text, raw_text))

    st.write(f"Resume uploaded: {len(resumes)}")


    if len(resumes) == 1:
        name, clean, raw = resumes[0]
        features = vectorizer.transform([clean])
        score = model.predict(features)[0]

        st.success(f"Overall resume ranking {name}: **{score:.2f}** / 5")
        with st.spinner("Explanation of the result from GPT-4o..."):
            explanation = explain_prediction(raw, score)
        st.markdown("#### Explanation of the model:")
        st.info(explanation)

    else:
        
        results = []
        for name, clean, raw in resumes:
            features = vectorizer.transform([clean])
            score = model.predict(features)[0]
            results.append((name, score, raw))

        
        results.sort(key=lambda x: x[1], reverse=True)

        st.subheader("Top candidates:")
        for idx, (name, score, raw) in enumerate(results, 1):
            st.markdown(f"**{idx}. {name}** — Рейтинг: **{score:.2f}** / 5")
            with st.expander("Explanation (GPT-4o)", expanded=False):
                with st.spinner("Generating an Explanation..."):
                    explanation = explain_prediction(raw, score)
                st.write(explanation)

        st.write("---")
