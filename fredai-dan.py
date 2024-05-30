import streamlit as st
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
from sklearn.preprocessing import LabelEncoder

@st.cache
def load_data():
    df = pd.read_csv('/content/test_danqa.csv')
    df.columns = ['question', 'answer']
    return df

@st.cache(allow_output_mutation=True)
def load_model():
    embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
    classifier = tf.keras.models.load_model("/content/trained_model.h5")  
    label_encoder = LabelEncoder()
    label_encoder.classes_ = np.load("/content/label_encoder_classes.npy")  
    return embed, classifier, label_encoder

@st.cache(hash_funcs={type(load_data): lambda _: None})
def load_data():
    df = pd.read_csv('/content/test_danqa.csv')
    df.columns = ['question', 'answer']
    return df

def embed_text(text):
    return embed(text).numpy()

def chatbot_response(user_input):
    user_embedding = embed_text([user_input])
    prediction = classifier.predict(user_embedding)
    predicted_label = prediction.argmax(axis=1)[0]
    return label_encoder.inverse_transform([predicted_label])[0]

st.title("Chatbot")
user_input = st.text_input("You:")
if st.button("Ask"):
    response = chatbot_response(user_input)
    st.text_area("Chatbot:", value=response, height=100)