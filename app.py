import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
import streamlit as st
from gtts import gTTS
import os

# Load your dataset (same one you used for training)
poetry_df = pd.read_csv('Roman-Urdu-Poetry.csv')  # Replace with your actual CSV file

# Initialize and fit the tokenizer
tokenizer = Tokenizer()
tokenizer.fit_on_texts(poetry_df['Poetry'])  # Assuming 'Poetry' is the column in your CSV

# Load the pre-trained model
model = load_model('poetry_model.h5')

# Function to generate poetry
def generate_poetry(model, tokenizer, start_word, num_words_per_line=10, num_lines=3, temperature=1.0, sampling_method='top_k', k=5, p=0.9):
    poem = start_word
    sequence = tokenizer.texts_to_sequences([start_word])
    sequence = pad_sequences(sequence, maxlen=model.input_shape[1], padding='pre')

    for line_num in range(num_lines):
        if line_num > 0:
            line = ""
        else:
            line = start_word
        
        for _ in range(num_words_per_line - 1):
            pred_probs = model.predict(sequence)[0]
            pred_probs = np.asarray(pred_probs).astype('float64')
            pred_probs = np.log(pred_probs) / temperature
            pred_probs = np.exp(pred_probs) / np.sum(np.exp(pred_probs))

            if sampling_method == 'top_k':
                predicted_word_index = top_k_sampling(pred_probs, k=k)
            elif sampling_method == 'top_p':
                predicted_word_index = top_p_sampling(pred_probs, p=p)
            else:
                predicted_word_index = np.random.choice(range(len(pred_probs)), p=pred_probs)

            predicted_word = tokenizer.index_word.get(predicted_word_index, "")

            if predicted_word == "":
                break

            line += ' ' + predicted_word
            sequence = np.append(sequence[:, 1:], predicted_word_index).reshape(1, -1)

        poem += '\n' + line

    return poem

def top_k_sampling(pred_probs, k=5):
    top_k_indices = np.argsort(pred_probs)[-k:]
    top_k_probs = pred_probs[top_k_indices]
    top_k_probs = top_k_probs / np.sum(top_k_probs)
    return np.random.choice(top_k_indices, p=top_k_probs)

def top_p_sampling(pred_probs, p=0.9):
    sorted_indices = np.argsort(pred_probs)[::-1]
    sorted_probs = pred_probs[sorted_indices]
    cumulative_probs = np.cumsum(sorted_probs)

    cutoff_index = np.where(cumulative_probs >= p)[0][0] + 1
    filtered_indices = sorted_indices[:cutoff_index]
    filtered_probs = sorted_probs[:cutoff_index]
    filtered_probs = filtered_probs / np.sum(filtered_probs)

    return np.random.choice(filtered_indices, p=filtered_probs)

st.title("Urdu Poetry Generator with Audio")
st.write("Generate poetry based on a starting word!")

start_word = st.text_input("Enter the starting word:", value="")
num_words_per_line = st.slider("Number of words per line:", min_value=5, max_value=20, value=10)
num_lines = st.slider("Number of lines:", min_value=1, max_value=5, value=3)
temperature = st.slider("Temperature (for creativity):", min_value=0.0, max_value=1.5, value=1.0)
sampling_method = st.selectbox("Sampling method:", options=["top_k", "top_p"])

# Define top-k and top-p parameters
k = st.slider("Top-k:", min_value=1, max_value=20, value=5)
p = st.slider("Top-p (nucleus sampling):", min_value=0.0, max_value=1.0, value=0.9)

# Generate poetry when button is clicked
if st.button("Generate Poetry"):
    if start_word:
        generated_poem = generate_poetry(model, tokenizer, start_word, num_words_per_line, num_lines, temperature, sampling_method, k, p)
        st.subheader("Generated Poetry:")
        st.text(generated_poem)

        # Text-to-Speech Conversion
        tts = gTTS(text=generated_poem, lang='ur', slow=False)
        audio_file = "poetry_audio.mp3"
        tts.save(audio_file)

        # Play the audio
        st.audio(audio_file)
    else:
        st.error("Please enter a starting word!")
