import subprocess
from ML_Pipeline import Train_Model
from ML_Pipeline.Preprocess import apply
from ML_Pipeline.Utils import load_model, save_model

# Ask the user for input (0 for training, 1 for prediction, and 2 for deployment)
val = int(input("Train - 0\nPredict - 1\nDeploy - 2\nEnter your value: "))

if val == 0:
    # Training mode
    x_train, y_train = apply("../input/review_data.csv", is_train=1)
    ml_model = Train_Model.fit(x_train, y_train)  # Train the model
    model_path = save_model(ml_model)  # Save the trained model
    print("Model saved in: ", "output/gru-model")

elif val == 1:
    # Prediction mode
    model_path = "../output/gru-model.h5"
    ml_model = load_model(model_path)  # Load the trained model
    x_test, y_test = apply("../input/test_review_data.csv", is_train=0)
    accuracy = ml_model.evaluate(x_test, y_test)[1] * 100.0
    print("Testing Accuracy: ", accuracy, "%")  # Evaluate the model on test data and print accuracy

else:
    # Deployment mode
    """
    process = subprocess.Popen(['python', 'ML_Pipeline/deploy.py'],
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE,
                               universal_newlines=True)
    """                           
    
    for stdout_line in process.stdout:
        print(stdout_line)

    stdout, stderr = process.communicate()
    print(stdout, stderr)
