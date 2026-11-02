# Python FAQ Chatbot Project

This project implements **TASK 2: Chatbot for FAQs** completely in Python using standard NLP libraries (`nltk`, `scikit-learn`, `cosine_similarity`).

---

## 🛠️ Requirements & Installation

Make sure you have Python installed, then install the required dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ How to Run

Run the chatbot script from your terminal:

```bash
python chatbot.py
```

---

## 🧠 Features Included:
1. **FAQ Collection**: Pre-loaded dictionary of questions and answers.
2. **Text Preprocessing**: Lowercasing, punctuation removal, tokenization, and stop-word removal using `NLTK`.
3. **Similarity Matching**: TF-IDF Vectorization combined with **Cosine Similarity** (`scikit-learn`) to score and match user queries against the FAQ database.
4. **Interactive CLI**: Command-line interface with similarity score logging and exit commands.
