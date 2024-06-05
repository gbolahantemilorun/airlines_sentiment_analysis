import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask, render_template, request, jsonify
from plotly.graph_objs import Bar
import joblib
from sqlalchemy import create_engine

app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    
    return clean_tokens

# Load data
DATABASE_URL = 'sqlite:///../data/AirlineSentiment.db'
engine = create_engine(DATABASE_URL)
df = pd.read_sql_table('tweets', engine)

# load model'
model = joblib.load("../models/classifier.pkl")

@app.route('/')
@app.route('/index')
def index():
    # Extract data needed for visuals
    sentiment_counts = df['airline_sentiment'].value_counts()
    sentiment_names = sentiment_counts.index.map({0: 'Negative', 1: 'Neutral', 2: 'Positive'})
    
    # Create visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=sentiment_names,
                    y=sentiment_counts
                )
            ],
            'layout': {
                'title': 'Distribution of Sentiments Used for Training',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Sentiment"
                }
            }
        }
    ]
    
    # Encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # Render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)

@app.route('/go')
def go():
    # Save user input in query
    query = request.args.get('query', '')
    
    # Use model to predict sentiment for query
    sentiment_prediction = model.predict([query])[0]
    sentiment_map = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
    sentiment_result = sentiment_map[sentiment_prediction]
    
    # Render the go.html with the sentiment result
    return render_template(
        'go.html',
        query=query,
        sentiment_result=sentiment_result
    )

def main():
    app.run(host='0.0.0.0', port=3001, debug=True)

if __name__ == '__main__':
    main()