import streamlit as st
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
import torch

@st.cache_resource
def load_data():
    tokenizer = AutoTokenizer.from_pretrained("./sentiment_model")
    model = AutoModelForSequenceClassification.from_pretrained("./sentiment_model")

    return tokenizer,model

tokenizer, model = load_data()

st.title("🎬 Movie Review Sentiment Analyzer")
st.write("Paste a movie review below and find out if it's positive or negative!")

review = st.text_area("Enter the review")
analyze = st.button("Analyze")

if analyze:
    inputs = tokenizer(review, padding="max_length", truncation=True, max_length=256, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    pred = torch.argmax(outputs.logits, dim=-1).item()

    if pred == 0:
        st.error("Negative review")
    else:
        st.success("Positive review")

