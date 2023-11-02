from flask import Flask, request
import Utils

app = Flask(__name)

# Load the pre-trained machine learning model
model_path = '../output/gru-model.h5'
ml_model = Utils.load_model(model_path)

@app.post("/get-review-score")
def get_review_score():
    from ML_Pipeline.Preprocess import apply_prediction

    # Receive POST requests with JSON data
    data = request.get_json()

    # Extract the review text from the JSON data
    review = data['review']

    # Use the loaded model to predict the review score
    prediction = apply_prediction(review, ml_model)

    # Prepare the prediction as JSON output
    output = {"Review Score": prediction}

    return output

if __name__ == '__main__':
    # Start the Flask app, making it accessible on host 0.0.0.0 (all available network interfaces) and port 5001
    app.run(host='0.0.0.0', port=5001)
