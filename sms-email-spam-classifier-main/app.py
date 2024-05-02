import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer
from scipy.sparse import csr_matrix


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

import pickle

# Load the TF-IDF vectorizer
with open('C:/Users/chaln/OneDrive/Desktop/PhishSleuth-main/sms-email-spam-classifier-main/vectorizer.pkl', 'rb') as file:
    tfidf = pickle.load(file)

# Load the machine learning model
try:
    with open('model.pkl', 'rb') as file:
        model = pickle.load(file)
except FileNotFoundError:
    print("Error: The file 'model.pkl' does not exist. Please make sure the file exists.")

model = pickle.load(open(r'C:\\Users\\chaln\\OneDrive\\Desktop\\PhishSleuth-main\\sms-email-spam-classifier-main\\model.pkl', 'rb'))


st.title("Email/SMS Spam Classifier")

input_sms = st.text_area("Enter the message")

if st.button('Predict'):

    # 1. preprocess
    transformed_sms = transform_text(input_sms)
    # 2. vectorize
    vector_input = tfidf.transform([transformed_sms])
    # 3. predict
    result = model.predict(vector_input)[0]
    # 4. Display
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")
