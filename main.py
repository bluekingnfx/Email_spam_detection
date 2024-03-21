import streamlit as st
import pickle
from sklearn.feature_extraction.text import CountVectorizer

import re
import nltk

nltk.download('stopwords')
from nltk.corpus import stopwords

# Load the trained model
model = pickle.load(open('naive_bayes.pkl', 'rb'))


# Text vectorization
with open('countvect.pkl', 'rb') as f:
    countvect = pickle.load(f)

def predict_email(text):
    stop_words = set(stopwords.words('english'))
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower()
    text = text.split()
    text = [w for w in text if w not in stop_words]
    text = ' '.join(text)
    x_test_df = countvect.transform([text])
    predicted_values_NB = model.predict(x_test_df)
    return predicted_values_NB[0]
    """ text = re.sub(r'[^a-zA-Z\s]', '', text)  
    text = text.lower() 
    text_list = [text] 

    email_data = countvect.transform(text_list)
    prediction = model.predict(email_data)
    return prediction """

def main():
    st.title("Spam Email Classifier")
    email_text = st.text_area("Enter the email text:", height=200)

    if st.button("Check"):
        if email_text:
            prediction = predict_email(email_text)
            st.write(f"The email is classified as: {prediction}")
        else:
            st.write("Please enter some text to classify.")

if __name__ == "__main__":
    main()