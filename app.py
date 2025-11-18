

import streamlit as st
import pickle
import string
import nltk
import os
# Set custom NLTK data path
nltk_data_path = os.path.join(os.getcwd(), "nltk_data")
nltk.data.path.append(nltk_data_path)

# Download required NLTK data to the custom folder
nltk.download("punkt", download_dir=nltk_data_path, quiet=True)
nltk.download("stopwords", download_dir=nltk_data_path, quiet=True)

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

st.title("Email Spam Classifier")

input_sms = st.text_area("Enter the message")

if st.button('Predict'):
    transformed_sms = transform_text(input_sms)
    vector_input = tfidf.transform([transformed_sms]).toarray()
    spam_proba = model.predict_proba(vector_input)[0][1]

    threshold = 0.4

    if spam_proba >= threshold:
        st.header("Spam")
    else:
        st.header("Not Spam")





