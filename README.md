# SentimentAI_Webapp

A web application for sentiment analysis using a RoBERTa transformer model. Enter any review or text, and the app will analyze its sentiment as Positive, Neutral, or Negative.

## Features

- Modern web UI for entering text and viewing results
- Uses [cardiffnlp/twitter-roberta-base-sentiment-latest](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest) model
- Real-time sentiment scoring via Flask backend

## Setup

1. **Clone the repository:**
   ```sh
   git clone <your-repo-url>
   cd SentimentAI_Webapp/src
   ```

2. **Install dependencies:**
   ```sh
   pip install flask transformers scipy torch
   ```

3. **Run the app:**
   ```sh
   python app.py
   ```

4. **Open your browser:**  
   Go to [http://127.0.0.1:5000](http://127.0.0.1:5000)

## Project Structure

```
src/
  app.py                  # Flask backend
  static/style.css        # Stylesheet
  templates/index.html    # Frontend HTML
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

