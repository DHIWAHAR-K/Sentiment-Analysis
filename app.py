import streamlit as st
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

def perform_sentiment_analysis(text, model, vectorizer):
    # Transform the text into numerical features
    text_features = vectorizer.transform([text])

    # Perform sentiment analysis on the text features
    sentiment = model.predict(text_features)[0]
    return sentiment

def main():
    # Load the pickled sentiment analysis model
    with open('C:/Users/adhit/OneDrive/Desktop/dhiwahar_k/Dhiwahar K Adhithya/College/YEAR 3/NLP/CA 4/moviereviews.pkl', 'rb') as file:
        model = pickle.load(file)

    # Load the pickled vectorizer
    with open('C:/Users/adhit/OneDrive/Desktop/dhiwahar_k/Dhiwahar K Adhithya/College/YEAR 3/NLP/CA 4/vectorizer.pkl', 'rb') as file:
        vectorizer = pickle.load(file)

    # Set the title and a brief description
    st.title("Sentiment Analysis")
    st.write("Enter some text and click the 'Analyze' button to get sentiment analysis results.")

    # Create a text input box for user input
    user_input = st.text_input("Enter text here")

    # Create a button to trigger the sentiment analysis
    if st.button("Analyze"):
        # Perform sentiment analysis on the user input
        sentiment = perform_sentiment_analysis(user_input, model, vectorizer)

        # Display the sentiment analysis results
        st.write(f"Sentiment: {sentiment}")

if __name__ == "__main__":
    main()
