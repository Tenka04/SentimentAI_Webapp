from flask import Flask, render_template, request, jsonify
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax

# Load RoBERTa model
MODEL = 'cardiffnlp/twitter-roberta-base-sentiment-latest'
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)

# Define function to predict sentiment
def polarity_scores(text):
    encoded = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
    output = model(**encoded)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    labels = ['Negative', 'Neutral', 'Positive']

    scores_dict = {label: float(score) for label, score in zip(labels, scores)}
    max_label = labels[scores.argmax()]
    scores_dict['label'] = f"This is {max_label}"

    return scores_dict

# Flask app
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json()
    text = data.get('text', '')
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    scores = polarity_scores(text)
    return jsonify(scores)

if __name__ == '__main__':
    print("Starting Flask app...")
    app.run(debug=True)
