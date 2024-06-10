# ChatBot

This repository contains a simple chatbot implemented in Python using TensorFlow and scikit-learn. The chatbot is trained on a set of intents defined in a JSON file and uses a neural network model to classify user input and respond appropriately.

## Prerequisites

Before you begin, ensure you have met the following requirements:

- Python 3.6 or later installed on your system
- Required Python packages installed (see below)

## Installation

1. Clone the repository to your local machine:

   ```sh
   git clone https://github.com/Akhil-Pothanapalli/chatbot_keras.git
   ```

2. Navigate to the project directory:

   ```sh
   cd chatbot
   ```

3. Install the required packages:

   ```sh
   pip install -r requirements.txt
   ```

## Files

- `intents.json`: Contains the training data for the chatbot, structured in intents.
- `chat_model`: Pre-trained model file.
- `tokenizer.pickle`: Tokenizer object file for preprocessing input data.
- `label_encoder.pickle`: Label encoder object file for transforming labels.

## Usage

1. Ensure you have the `intents.json`, `chat_model`, `tokenizer.pickle`, and `label_encoder.pickle` files in the project directory.

2. Run the chatbot:

   ```sh
   python chatbot.py
   ```

3. Start interacting with the bot. Type your messages and press Enter. Type `quit` to stop the interaction.

## Code Overview

### Loading Required Libraries

```python
import json
import numpy as np
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder
import colorama 
colorama.init()
from colorama import Fore, Style, Back
import random
import pickle
```

### Loading Data

The intents are loaded from a JSON file:

```python
with open("intents.json") as file:
    data = json.load(file)
```

### Chat Function

The main `chat` function loads the trained model, tokenizer, and label encoder. It then enters an interactive loop where it accepts user input, processes it, and prints the chatbot's response:

```python
def chat():
    model = keras.models.load_model('chat_model')
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    with open('label_encoder.pickle', 'rb') as enc:
        lbl_encoder = pickle.load(enc)
    max_len = 20
    
    while True:
        print(Fore.LIGHTBLUE_EX + "User: " + Style.RESET_ALL, end="")
        inp = input()
        if inp.lower() == "quit":
            break

        result = model.predict(keras.preprocessing.sequence.pad_sequences(tokenizer.texts_to_sequences([inp]),
                                             truncating='post', maxlen=max_len))
        tag = lbl_encoder.inverse_transform([np.argmax(result)])

        for i in data['intents']:
            if i['tag'] == tag:
                print(Fore.GREEN + "ChatBot:" + Style.RESET_ALL , np.random.choice(i['responses']))

print(Fore.YELLOW + "Start messaging with the bot (type quit to stop)!" + Style.RESET_ALL)
chat()
```

## Acknowledgements

- TensorFlow
- scikit-learn
- Colorama

- Thank you Amila Viraj and I his medium post helped me in this learning journey - https://towardsdatascience.com/how-to-build-your-own-chatbot-using-deep-learning-bb41f970e281
