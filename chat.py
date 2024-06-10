import json
import numpy as np
import tensorflow as tf
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize
import random

# Load metadata
try:
    with open('metadata.json', 'r') as f:
        metadata = json.load(f)
except Exception as e:
    print(f"Error loading metadata: {e}")
    exit(1)

input_size = metadata["input_size"]
hidden_size = metadata["hidden_size"]
output_size = metadata["output_size"]
all_words = metadata["all_words"]
tags = metadata["tags"]
label_encoder = {i: tag for i, tag in enumerate(metadata["label_encoder"])}

# Load the model
try:
    model = tf.keras.models.load_model("chat_model")
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

bot_name = "Sam"
print("Let's chat! (type 'quit' to exit)")
while True:
    sentence = input("You: ")
    if sentence.lower() == "quit":
        break

    # Preprocess user input
    bow = bag_of_words(tokenize(sentence), all_words)
    bow = np.array([bow])

    # Predict intent
    results = model(bow)
    predicted = np.argmax(results, axis=1)
    tag = label_encoder[predicted[0]]

    # Load the intents file to get responses
    with open('intents.json', 'r') as json_data:
        intents = json.load(json_data)

    for intent in intents['intents']:
        if tag == intent["tag"]:
            print(f"{bot_name}: {random.choice(intent['responses'])}")
