import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
import pickle

# Load the pre-trained model
with open('C:\\Users\\divya mulchandani\\OneDrive\\Documents\\AI-assignment\\box office\\box office\\module\\model1.pkl', 'rb') as f:
    model_dict = pickle.load(f)
    gb_model = model_dict['model']

# Define a function for preprocessing input data
def preprocess_input(movie_name, data):
    if movie_name in data.index:
        features = data.loc[movie_name, ['facenumber_in_poster', 'director_ig_followers', 'actor_1_ig_follow', 'actor_2_ig_follow', 'duration', 'buget', 'gross', 'num_critic_for_review', 'num_user_for_review', 'imdb_socre', 'num_awards_won']].tolist()
        while len(features) < 16:
            features.append(0)
        return features
    else:
        return None

# Load the dataset
data = pd.read_csv('C:\\Users\\divya mulchandani\\OneDrive\\Documents\\AI-assignment\\box office\\box office\\box_office.csv', encoding='latin-1')
data.set_index('movie_title', inplace=True)

# Streamlit app
st.title('Box Office Revenue Prediction')

movie_name = st.text_input('Enter movie name:')

if st.button('Predict'):
    input_data = preprocess_input(movie_name, data)
    if input_data is None:
        st.error(f"Movie '{movie_name}' not found in the dataset.")
    else:
        predicted_box_office = gb_model.predict([input_data])[0]
        st.success(f"Predicted box office revenue for '{movie_name}': {predicted_box_office:.2f}")
