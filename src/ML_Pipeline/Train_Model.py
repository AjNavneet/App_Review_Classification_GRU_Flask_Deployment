from keras.layers import Dense, Embedding, GRU, SpatialDropout1D
from tensorflow.keras.models import Sequential
from ML_Pipeline import Utils

# Function to train the ML model
def train(model, x_train, y_train):
    batch_size = 32
    epochs = 50
    history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose='auto')
    return model

# Function to initiate the model and training data
def fit(x_train, y_train):
    # Create a sequential model
    model = Sequential()
    
    # Add an embedding layer
    model.add(Embedding(Utils.input_length, 120, input_length=x_train.shape[1]))
    
    # Add spatial dropout
    model.add(SpatialDropout1D(0.4))
    
    # Add a GRU (Gated Recurrent Unit) layer
    model.add(GRU(64, dropout=0.2, recurrent_dropout=0.2))
    
    # Add a dense output layer with softmax activation for classification
    model.add(Dense(5, activation='softmax'))
    
    # Compile the model with loss, optimizer, and metrics
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    print(model.summary())

    # Train the model
    model = train(model, x_train, y_train)

    return model
