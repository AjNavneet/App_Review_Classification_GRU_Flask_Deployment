# App Review Classification Model using Gated Recurrent Unit (GRU) and Flask Deployment

## Overview

Gated Recurrent Unit (GRU) is an enhanced version of the standard Recurrent Neural Network (RNN) that addresses the problem of short-term memory and is well-suited for sequential data processing. This project focuses on building a review classification model using GRU to classify app reviews on a scale of 1 to 5, with 1 indicating negative sentiment and 5 indicating positive sentiment.

---

### Aim

To build a classifier to classify app reviews on a scale of 1 to 5 using Gated Recurrent Unit (GRU).

---

### Data Description

The dataset consists of app reviews and their corresponding ratings. The "score" column contains ratings ranging from 1 to 5 based on the "content" column.

---

### Tech Stack

- Language: `Python`
- Libraries: `pandas`, `TensorFlow`, `Matplotlib`, `scikit-learn`, `Pillow`, `Gunicorn`, `TextBlob`, `NLTK`, `Keras`, `Flask`

---

## Approach

### Data Preprocessing

1. Converting words to lowercase.
2. Lemmatization of words.
3. Tokenization of words.
4. One-hot encoding of the scores.

### Model Training

- Training a sequential model in TensorFlow.

### Model Evaluation

- Evaluating the model on test data.

---

## Modular Code Overview

1. **Input**: Contains the data used for analysis, including:
   - `review_data.csv`
   - `test_review_data.csv` (for testing)

2. **Output**: Contains the saved GRU model and pickle file.

3. **src**: The heart of the project, this folder contains all the modularized code for various steps. It further includes:
   - **ML_pipeline**: A folder with functions organized into different Python files, each appropriately named for its functionality. These Python functions are called inside the `Engine.py` file.

1. **GRU-Neural-Network.ipynb** and **Model_Api.ipynb**: The original Python notebooks.

2. **requirements.txt**: Lists all the required libraries and their versions for easy installation using `pip`.

---

