# app.py - Final Streamlit Application Script

import streamlit as st
import tensorflow as tf
import pickle
import re
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os

# --- Page Configuration ---
st.set_page_config(
    page_title="Project DisasterTweet",
    page_icon="🛡️",
    layout="centered",
    initial_sidebar_state="auto"
)

# --- Load Model and Tokenizer ---
# Use a caching mechanism to load the model and tokenizer only once.
@st.cache_resource
def load_assets():
    """Loads the trained model and tokenizer from the assets folder."""
    model_path = os.path.join('assets', 'project_guardian_model_final.h5')
    tokenizer_path = os.path.join('assets', 'tokenizer.pickle')

    try:
        model = tf.keras.models.load_model(model_path)
        with open(tokenizer_path, 'rb') as handle:
            tokenizer = pickle.load(handle)
        return model, tokenizer
    except Exception as e:
        st.error(f"Error loading model assets: {e}")
        return None, None


model, tokenizer = load_assets()


# --- Text Preprocessing Function ---
def clean_text(text):
    """Cleans the input text using the same steps as in training."""
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'@[A-Za-z0-9_]+', '', text)
    text = re.sub(r'#', '', text)
    text = re.sub(r'[^a-z0-9\s]', '', text) # <-- IMPORTANT CHANGE: Now keeps numbers (0-9)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# --- Streamlit UI ---
st.title("🛡️ Project DisasterTweet")
st.subheader("Disaster Response - Tweet Classifier")
st.markdown("""
This tool uses a Deep Learning model to analyze a tweet and predict whether it's related to a real disaster. 
This helps filter through noise and identify urgent calls for help during a crisis.
""")

st.markdown("---")

# Default example text
example_text = "My house is on fire, we are trapped upstairs please send help to 123 Main St"

# User input text area
user_input = st.text_area("Enter a tweet to classify:", example_text, height=100)

if st.button('Analyze Tweet'):
    if user_input and model and tokenizer:
        # 1. Clean and preprocess the input
        cleaned_input = clean_text(user_input)
        sequence = tokenizer.texts_to_sequences([cleaned_input])
        padded_sequence = pad_sequences(sequence, maxlen=30, padding='post', truncating='post')

        # 2. Make a prediction
        prediction = model.predict(padded_sequence)[0][0]
        is_disaster = prediction > 0.5  # Threshold is 0.5

        # 3. Display the result with a progress bar and detailed info
        st.markdown("---")
        st.write("### Analysis Result")

        # Display a progress bar for visual effect
        progress_bar = st.progress(0)
        progress_text = st.empty()

        if is_disaster:
            progress_bar.progress(int(prediction * 100), text=f"Confidence: {prediction * 100:.0f}%")
            st.error("**Classification: REAL DISASTER**")
            st.warning(
                "This tweet appears to be a genuine report about a real-time, ongoing disaster. Action may be required.",
                icon="🚨")
        else:
            progress_bar.progress(int((1 - prediction) * 100), text=f"Confidence: {(1 - prediction) * 100:.0f}%")
            st.success("**Classification: NOT A DISASTER**")
            st.info("This tweet does not appear to be a direct report of a disaster.", icon="✅")

    elif not model or not tokenizer:
        st.error("Model assets could not be loaded. Please check the console for errors.")
    else:
        st.warning("Please enter a tweet to analyze.")

st.markdown("---")
st.markdown("Built by Surya Nallaparaju. This project demonstrates the application of Deep Learning for social good.")