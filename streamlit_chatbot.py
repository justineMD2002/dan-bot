!pip install pandas tensorflow tensorflow-hub tensorflow-text scikit-learn

import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

df = pd.read_csv('/content/test_danqa.csv')
df.columns = ['question', 'answer']

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df['answer'])

X_train, X_val, y_train, y_val = train_test_split(df['question'], y, test_size=0.2, random_state=42)

embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

classifier = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(512,)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.4), 
    tf.keras.layers.Dense(len(label_encoder.classes_), activation='softmax')
])

classifier.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

X_train_embed = embed(X_train.tolist())
X_val_embed = embed(X_val.tolist())

classifier.fit(X_train_embed, y_train, epochs=15, batch_size=8,  
               validation_data=(X_val_embed, y_val), verbose=0)

def chatbot_response(user_input):
    user_embedding = embed([user_input])
    prediction = classifier.predict(user_embedding)
    predicted_label = prediction.argmax(axis=1)[0]
    return label_encoder.inverse_transform([predicted_label])[0]