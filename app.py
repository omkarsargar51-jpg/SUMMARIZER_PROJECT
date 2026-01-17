import streamlit as st
from transformers import pipeline

st.title("AI Text Summarizer")

# Load model
@st.cache_resource
def load_model():
    return pipeline("summarization")

summarizer = load_model()

# Input
text = st.text_area("Enter text to summarize:", height=200)

if st.button("Summarize"):
    if text:
        with st.spinner("Summarizing..."):
            result = summarizer(text, max_length=130, min_length=30)
            st.success(result[0]['summary_text'])