# 🌏 Project DisasterTweet: Real-Time Tweet Classifier

**Live Demo:** https://projectdisastertweet.streamlit.app

This project is an end-to-end deep learning solution for classifying whether tweets are about real disasters or not. The core of the project is a Bidirectional LSTM model, which is deployed as an interactive web application using Streamlit.

### Key Features

*   **Deep Learning Model:** Utilizes a stacked Bi-LSTM architecture with pre-trained GloVe embeddings for high-accuracy text classification.
*   **End-to-End Pipeline:** Covers the complete process from data cleaning and preprocessing to model training and evaluation.
*   **Interactive Web App:** A user-friendly interface built with Streamlit that allows for real-time tweet classification.
*   **Performance Metrics:** The model achieves **80.3% validation accuracy** and a **0.76 F1-score**.

### Tech Stack

*   **Language:** Python
*   **Libraries:** TensorFlow (Keras), Scikit-learn, Pandas, NLTK, Streamlit
*   **Embeddings:** GloVe (Global Vectors for Word Representation)
*   **Dataset:** [Natural Language Processing with Disaster Tweets](https://www.kaggle.com/competitions/nlp-getting-started/overview)
