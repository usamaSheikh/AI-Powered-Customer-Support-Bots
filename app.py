# libraries
import random
import numpy as np
import pickle
import json
from flask import Flask, render_template, request, jsonify
import nltk
from nltk.stem import WordNetLemmatizer
from keras.models import load_model

# Ensure NLTK can find the data files
nltk.data.path.append('C:/nltk_data')  # Make sure this path matches where your nltk_data folder is located
nltk.download('punkt')
nltk.download('wordnet')

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Load model and data
try:
    model = load_model("chatbot_model.h5")
    intents = json.loads(open("intents.json").read())
    print("Intents JSON loaded successfully:", intents)  # Debug print to confirm JSON loaded
    words = pickle.load(open("words.pkl", "rb"))
    classes = pickle.load(open("classes.pkl", "rb"))
    print("Model and data files loaded successfully.")
except Exception as e:
    print(f"Error loading model or data files: {e}")

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get", methods=["POST"])
def chatbot_response():
    try:
        msg = request.form.get("msg")
        if not msg:
            return jsonify({"error": "No message received"}), 400

        print("Received message:", msg)  # Debugging print

        if msg.startswith('my name is'):
            name = msg[11:]
            ints = predict_class(msg, model)
            res1 = getResponse(ints, intents)
            res = res1.replace("{n}", name)
        elif msg.startswith('hi my name is'):
            name = msg[14:]
            ints = predict_class(msg, model)
            res1 = getResponse(ints, intents)
            res = res1.replace("{n}", name)
        else:
            ints = predict_class(msg, model)
            res = getResponse(ints, intents)

        print("Generated response:", res)  # Debugging print
        return jsonify({"response": res})

    except Exception as e:
        print(f"Error processing message: {e}")
        return jsonify({"error": "Internal server error"}), 500

# chat functionalities
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence.lower())  # Convert to lowercase
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    print("Tokenized and lemmatized words:", sentence_words)  # Debug print
    return sentence_words

def bow(sentence, words, show_details=True):
    sentence_words = clean_up_sentence(sentence)
    print("Tokenized words:", sentence_words)  # Debug print
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print("found in bag: %s" % w)
    print("Bag of words:", bag)  # Debug print
    return bag

def predict_class(sentence, model):
    try:
        # Generate bag of words
        p = bow(sentence, words, show_details=False)

        # Ensure the bag of words matches the expected input size of the model
        expected_length = 87
        if len(p) > expected_length:
            p = p[:expected_length]  # Truncate if it's too long
        elif len(p) < expected_length:
            p += [0] * (expected_length - len(p))  # Pad with zeros if it's too short

        # Convert to numpy array and reshape to fit model input
        p = np.array([p])
        print("Processed bag of words for sentence (adjusted):", p)  # Debug print

        res = model.predict(p)[0]
        ERROR_THRESHOLD = 0.1
        results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
        
        results.sort(key=lambda x: x[1], reverse=True)
        return_list = []
        for r in results:
            return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
        
        print("Predicted intents:", return_list)  # Debug print
        return return_list
    except Exception as e:
        print(f"Error in predict_class: {e}")
        return []

def getResponse(ints, intents_json):
    if not ints:  # Check if `ints` is empty
        return "I'm sorry, I didn't understand that."

    tag = ints[0]["intent"]
    list_of_intents = intents_json["intents"]
    for i in list_of_intents:
        if i["tag"] == tag:
            result = random.choice(i["responses"])
            break
    print("Selected response:", result)  # Debug print
    return result

if __name__ == "__main__":
    app.run(debug=True)
