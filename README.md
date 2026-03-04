# 🎬 Movie Review Sentiment Analyzer

A fine-tuned DistilBERT model for binary sentiment classification on movie reviews, served via an interactive Streamlit web application.

---

## Overview

This project fine-tunes a pre-trained DistilBERT transformer on the IMDB dataset (25,000 movie reviews) to classify reviews as positive or negative. The trained model is served through a Streamlit web app where users can paste any movie review and receive an instant sentiment prediction.

This is not a wrapper around an existing sentiment API — the model was trained from scratch on labeled data using the HuggingFace Trainer API.

---

## Results

| Metric | Score |
|--------|-------|
| Accuracy | ~91% |
| F1 Score | ~91% |
| Dataset | IMDB (25,000 train / 25,000 test) |
| Base Model | distilbert-base-uncased |
| Training Time | ~25 mins on NVIDIA T4 GPU |

---

## Tech Stack

- **Model:** DistilBERT (distilbert-base-uncased) via HuggingFace Transformers
- **Dataset:** IMDB via HuggingFace Datasets
- **Training:** HuggingFace Trainer API with PyTorch backend
- **Evaluation:** scikit-learn (accuracy, F1)
- **Web App:** Streamlit
- **Hardware:** Google Colab (NVIDIA T4 GPU)

---

## Project Structure

```
sentiment-analyzer/
│
├── train.py              # Fine-tuning script
├── app.py                # Streamlit web application
├── sentiment_model/      # Saved model weights and tokenizer (see note below)
└── README.md
```

> **Note:** Model weights are not included in this repository due to file size (~250MB). Download the trained model from [HuggingFace](https://huggingface.co/devansh7733/Sentiment-analyzer) and place it in a `sentiment_model/` folder in the project root.

---

## How to Run

**1. Clone the repository**
```bash
git clone https://github.com/yourusername/sentiment-analyzer
cd sentiment-analyzer
```

**2. Install dependencies**
```bash
pip install transformers datasets torch scikit-learn streamlit accelerate
```

**3. Download the model**

Download the trained model weights from [HuggingFace](https://huggingface.co/devansh7733/Sentiment-analyzer) and place the files in a folder called `sentiment_model/` in the project root.

**4. Run the app**
```bash
streamlit run app.py
```

The app will open automatically at `http://localhost:8501`.

**To retrain from scratch:**
```bash
python train.py
```
> Note: Training on the full IMDB dataset is recommended on a GPU. On CPU this will take several hours.

---

## Limitations and Known Issues

**Domain shift:** The model was trained exclusively on English-language Hollywood film reviews from IMDB. Performance degrades noticeably on:
- Reviews written in non-English languages or mixed-language styles
- Reviews of non-Western films with different cultural context
- Reviews with complex mixed sentiment (negative opening, positive conclusion)

**Truncation:** Reviews longer than 256 tokens are truncated. In rare cases where sentiment is established late in a review, this may affect prediction accuracy.

**Potential improvement:** Fine-tuning on a more diverse, multilingual review dataset would likely improve robustness across these edge cases.

---

## Key Takeaways

- How transformer self-attention captures contextual meaning that classical bag-of-words approaches miss
- The fine-tuning workflow — loading pre-trained weights, attaching a classification head, running a training loop
- Why overfitting occurs on small datasets and how to detect it from validation metrics
- The difference between accuracy and F1 score and when each metric matters
- How to identify and honestly evaluate domain shift failures in a deployed model

---

## Acknowledgements

- [HuggingFace](https://huggingface.co) for the Transformers and Datasets libraries
- [IMDB Dataset](https://huggingface.co/datasets/imdb) — Maas et al., 2011
- [DistilBERT](https://arxiv.org/abs/1910.01108) — Sanh et al., 2019
