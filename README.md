# Admission Chatbot Documentation

## Overview

The Admission Chatbot is a project that implements a chatbot to assist users with inquiries related to admissions at IIT BHU. The chatbot is trained on a dataset provided in the `intents.json` file, using the NLTK library for transfer learning. The project includes components for natural language processing, machine learning model training, and chatbot interaction.

## Files


1. **nltk_utils.py**
   - Contains utility functions for tokenization, stemming, and bag-of-words representation using NLTK.

2. **model.py**
   - Defines the structure of the neural network model using PyTorch.

3. **intents.json**
   - JSON file containing intents, patterns, and responses for training the chatbot.

4. **data.pth**
   - Saved PyTorch model state and other necessary information.

5  **train.py**
  - contains code for training nltk model on the specified queries of admission.

## Prerequisites

- Python 3.x
- Tensorflow
- NLTK

## Installation

1. Clone the repository:

   ```bash
   git clone https://https://github.com/Zsclarx/Project-2-Basic-Q-A-Bot-for-College-Admission
   ```

2. Install dependencies:

   ```bash
   pip install torch nltk
   ```

3. Run the chatbot script:

   ```bash
   python train.py
   python chat.py
   ```

## Training the Model

The model is trained on the intents provided in `intents.json`. The training script (`train_model.py`) processes the data, tokenizes sentences, and trains the neural network using PyTorch.

```bash
python train.py
```

## Usage

1. Input queries or questions related to IIT BHU admissions.

2. The chatbot will respond with appropriate answers based on the trained model.

## Model Details

- The neural network model architecture is defined in `model.py`.
- The model is trained using cross-entropy loss and the Adam optimizer.
- Training hyperparameters such as epochs, batch size, and learning rate can be adjusted in the script.

## Intent Categories

The `intents.json` file contains different intent categories such as greetings, goodbyes, admission-related queries, scholarship information, and more. Each intent has associated patterns and responses for training the chatbot.
This dataset is used for finer tuning of the model.

## Contact

For further assistance or inquiries, please contact:

- [Prakhar Pratap Singh]
- [prakhar2704@gmail.com]

---
