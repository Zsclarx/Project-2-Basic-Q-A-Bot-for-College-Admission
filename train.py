import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize, stem
import json
# Load the CSV data
data = pd.read_csv('intents.csv')

# Preprocess the data
all_words = []
tags = data['tag'].unique()
patterns = data['patterns']

for pattern in patterns:
    words = tokenize(pattern)
    all_words.extend(words)

ignore_words = ['?', '!', '.', ',']
all_words = [stem(w) for w in all_words if w not in ignore_words]
all_words = sorted(set(all_words))

X_train = [bag_of_words(tokenize(pattern), all_words) for pattern in patterns]
X_train = np.array(X_train)

# Encode labels
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(data['tag'])

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

# Create TensorFlow datasets
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(len(X_train)).batch(8)
val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(8)

# Define model parameters
input_size = len(X_train[0])
hidden_size = 128
output_size = len(tags)

# Initialize the model
model = NeuralNet(input_size, hidden_size, output_size)

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['accuracy'])

# Train the model
model.fit(train_dataset, epochs=200, validation_data=val_dataset)

# Save the model in TensorFlow SavedModel format
model.save("chat_model")

# Save metadata
metadata = {
    "input_size": input_size,
    "hidden_size": hidden_size,
    "output_size": output_size,
    "all_words": all_words,
    "tags": list(tags),
    "label_encoder": label_encoder.classes_.tolist()
}

with open("metadata.json", "w") as f:
    json.dump(metadata, f)
