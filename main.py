from flask import Flask, request, jsonify
import pickle
import pandas as pd

app = Flask(__name__)

# Load the pickled model
model_path = r"C:\Users\wanji\Downloads\random_forest_model.pkl"

with open(model_path,'rb') as file:
    model = pickle.load(file)

# Create the 'Hour' column from the TimeStamp
def create_hour_column(df):
    data_2['TimeStamp'] = pd.to_datetime(data_2['TimeStamp'])
    data_2['Hour'] = data_2['TimeStamp'].dt.hour
    return data_2
# Define the route for the prediction
@app.route('/')
def index():
    return 'Index Page'


@app.route('/predict', methods=['POST'])
def predict():
    content = request.get_json()
    # Load the data from the csv
    csv_file = r"C:\Users\wanji\Downloads\C__Users_wanji_Desktop_Green speed notebook_Green-Speed_data.csv"
    data_2= pd.read_csv(csv_file)

    # Process the data: Fill in the missing values
    data_2 = data_2[['Hour','Traffic Concentration']]

    # Make predictions using the loaded model
    predictions = model.predict(data_2)
    # Get instances where the prediction is "Eco-Friendly"
    eco_friendly_data = []
    for i in range(len(data_2)):
        if predictions[i] == 1:  # Check if the model predicted "Eco-Friendly"
            hour = data_2.iloc[i]['Hour']
            traffic_flow = data_2.iloc[i]['Traffic Concentration']
            eco_friendly_data.append({
                "Hour": hour,
                "TrafficFlow": traffic_flow,
                "Prediction": "Eco-Friendly"
            })

    # Return the predictions and structured output as JSON response
    return jsonify({'predictions': eco_friendly_data})

if __name__ =='__main__':
    app.run(debug= True)



    



