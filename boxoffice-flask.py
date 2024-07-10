from flask import Flask, render_template, request
import pickle
import pandas as pd
from sklearn.impute import SimpleImputer

app = Flask(__name__)

# Load the trained Gradient Boosting Regressor model
model_path = 'C:\\Users\\divya mulchandani\\OneDrive\\Documents\\AI-assignment\\box office\\box office\\module\\gb_model.pkl'
with open(model_path, "rb") as file:
    model = pickle.load(file)

# Load the dataset to fetch actual box office values
data_path = 'C:\\Users\\divya mulchandani\\OneDrive\\Documents\\AI-assignment\\box office\\box office\\box_office.csv'
data = pd.read_csv(data_path, encoding='latin-1')
data.set_index('movie_title', inplace=True)

# Define function to preprocess input data
def preprocess_input(movie_name, data):
    # Check if the provided movie name exists in the dataset
    if movie_name in data.index:
        # Extract features corresponding to the movie name from the dataset
        features = data.loc[movie_name, ['facenumber_in_poster', 'director_ig_followers', 'actor_1_ig_follow', 'actor_2_ig_follow', 'duration', 'buget', 'gross', 'num_critic_for_review', 'num_user_for_review', 'imdb_socre', 'num_awards_won']].tolist()
        # Adjust the number of features to match the model's expectations (if needed)
        while len(features) < 16:
            features.append(0)  # Add zeros to match the expected number of features
        return features
    else:
        # If the movie name is not found in the dataset, return None
        return None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        movie_name = request.form['movie_name']
        # Preprocess input data
        input_data = preprocess_input(movie_name, data)
        if input_data is None:
            return render_template('error.html', message=f"Movie '{movie_name}' not found in the dataset.")
        
        # Handle missing values by imputing
        imputer = SimpleImputer(strategy='mean')
        input_data_imputed = imputer.fit_transform([input_data])
        
        # Make prediction
        predicted_box_office = model.predict(input_data_imputed)[0]
        
        # Fetch actual box office value for the movie
        actual_box_office = data.loc[movie_name, 'box_office']
        
        return render_template('result.html', movie_name=movie_name, predicted=predicted_box_office, actual=actual_box_office)

if __name__ == '__main__':
    app.run(debug=True)
