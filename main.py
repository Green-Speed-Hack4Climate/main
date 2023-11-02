from flask import Flask, request, jsonify
import pickle
import pandas as pd

app = Flask(__name__)

# Load the pickled model
model_path = r"C:\Users\wanji\Downloads\random_forest_model.pkl"

with open(model_path,'rb') as file:
    model = pickle.load(file)
# Define the route for the prediction
@app.route('/')
def index():
    return 'Index Page'


@app.route('/predict', methods=['POST'])
def predict():
    content = request.get_json()
    # Load the data from the csv
    csv_file = "C:\\Users\\wanji\\Desktop\\Grren Speed\\datexDataA13 clean.csv"
    data_2= pd.read_csv(csv_file)

    # Process the data: Fill in the missing values
    data_2 = data_2[['Hour','Traffic Concentration']]

    # Fill the missing values
    data_2['Average Vehicle Speed'] = data_2['Average Vehicle Speed'].fillna(data_2['Average Vehicle Speed'].mean())

    # Make predictions using the loaded model
    predictions = model.predict(data_2)

    # Return the predictions as JSON request
    return jsonify({'predictions': predictions.tolist()})

if __name__ =='__main__':
    app.run(debug= True)



    



