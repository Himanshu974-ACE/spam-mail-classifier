import streamlit as st
import pickle
import string
import nltk
import re

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# ---------- NLTK + STOPWORDS SETUP ----------
ps = PorterStemmer()

# Try to load stopwords once and store them in a Python set
try:
    stop_words = set(stopwords.words('english'))
except LookupError:
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))


# ---------- TEXT PREPROCESSING ----------
def transform_text(text):
    # 1) lower-case
    text = text.lower()

    # 2) simple tokenization (regex – no punkt_tab issue)
    tokens = re.findall(r'\b\w+\b', text)

    # 3) keep only alpha-numeric tokens
    cleaned_tokens = []
    for token in tokens:
        if token.isalnum():
            cleaned_tokens.append(token)

    # 4) remove stopwords and punctuation
    filtered_tokens = []
    for token in cleaned_tokens:
        if token not in stop_words and token not in string.punctuation:
            filtered_tokens.append(token)

    # 5) stemming
    stemmed_tokens = []
    for token in filtered_tokens:
        stemmed_tokens.append(ps.stem(token))

    return " ".join(stemmed_tokens)


# ---------- LOAD MODEL & VECTORIZER ----------
# Make sure these are the SAME files you saved after final training.
tfidf = pickle.load(open("vectorizer.pkl", "rb"))
model = pickle.load(open("model.pkl", "rb"))


# ---------- STREAMLIT UI ----------
st.title("Email Spam Classifier")

input_sms = st.text_area("Enter the message")

if st.button("Predict"):
    if not input_sms.strip():
        st.warning("Please enter a message first.")
    else:
        # 1. Preprocess
        transformed_sms = transform_text(input_sms)

        # 2. Vectorize
        vector_input = tfidf.transform([transformed_sms])

        # 3. Predict
        result = model.predict(vector_input)[0]

        # (Optional) See probabilities for debugging
        # if hasattr(model, "predict_proba"):
        #     proba = model.predict_proba(vector_input)[0]
        #     st.write(f"Spam probability: {proba[1]:.2f}")

        # 4. Show result
        if result == 1:
            st.error("This message looks like **SPAM**.")
        else:
            st.success("This message looks **NOT SPAM**.")
