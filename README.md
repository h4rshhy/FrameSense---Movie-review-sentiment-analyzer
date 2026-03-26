# Sentiment Analysis Project

This project implements a sentiment analysis model using a simple structure with a backend and frontend. The backend handles the sentiment analysis logic, while the frontend provides a user interface for interaction.

## Project Structure

```
sentiment-analysis
├── src
│   ├── backend.py
│   └── frontend.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Setup Instructions

1. **Clone the repository:**
   ```
   git clone <repository-url>
   cd sentiment-analysis
   ```

2. **Create a virtual environment:**
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install the required dependencies:**
   ```
   pip install -r requirements.txt
   ```

## Usage

1. **Run the backend server:**
   ```
   python src/backend.py
   ```

2. **Open the frontend interface:**
   ```
   python src/frontend.py
   ```

3. **Input text in the provided field and submit to see the sentiment analysis results.**

## Dependencies

- Flask
- NLTK
- TextBlob
- transformers

## License

This project is licensed under the MIT License.