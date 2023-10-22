def init():
    global model, preprocessor
    model_path = Model.get_model_path(model_name='best_knn_model')  # Model name
    model = joblib.load(model_path)
    preprocessor = joblib.load('preprocessor.pkl')  # Load preprocessor 

# Perform scoring using the loaded model and preprocessor
def run(raw_data):
    try:
        data = json.loads(raw_data)
        # Assuming 'features' is a list of dictionaries where each dictionary corresponds to a data point

        # Extract features from the input data
        features = data['features']

        # Apply the preprocessing pipeline to the input data
        preprocessed_data = preprocessor.transform(features)

        # Use the loaded model for predictions
        predictions = model.predict(preprocessed_data)

        # Return the predictions as a JSON object
        return json.dumps({"predictions": predictions.tolist()})
    except Exception as e:
        error = str(e)
        return json.dumps({"error": error})
