import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import re
import nltk
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download NLTK data files
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def load_data(database_filepath):
    """
    Load data from SQLite database.

    Args:
    database_filepath: string. Filepath for the SQLite database.

    Returns:
    X: pandas DataFrame. Features dataset.
    Y: pandas DataFrame. Target dataset.
    """
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table('tweets', engine)
    X = df['text']
    Y = df['airline_sentiment']
    return X, Y

def tokenize(text):
    """
    Tokenize and clean text.

    Args:
    text: string. Text data to be tokenized.

    Returns:
    clean_tokens: list of strings. List of clean tokenized words.
    """
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stopwords.words("english")]
    lemmatizer = WordNetLemmatizer()
    clean_tokens = [lemmatizer.lemmatize(tok).lower().strip() for tok in tokens]
    return clean_tokens

def build_model():
    """
    Build machine learning pipeline and perform grid search.

    Returns:
    cv: GridSearchCV object. Grid search model object.
    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),  # Convert text to word count vectors
        ('tfidf', TfidfTransformer()),  # Convert word count vectors to TF-IDF representation
        ('clf', RandomForestClassifier())  # Classifier using RandomForest
    ])

    parameters = {
        'clf__n_estimators': [50, 100],
        'clf__min_samples_split': [2, 5]
    }

    cv = GridSearchCV(pipeline, param_grid=parameters, verbose=3, n_jobs=1)
    return cv

def evaluate_model(model, X_test, Y_test):
    """
    Evaluate model performance and print classification report.

    Args:
    model: GridSearchCV object. Trained model object.
    X_test: pandas DataFrame. Test features.
    Y_test: pandas DataFrame. Test targets.
    """
    Y_pred = model.predict(X_test)
    print(classification_report(Y_test, Y_pred, target_names=['negative', 'neutral', 'positive']))
    accuracy = (Y_pred == Y_test).mean()
    print("Overall Accuracy:", accuracy)

def save_model(model, model_filepath):
    """
    Save the model as a pickle file.

    Args:
    model: GridSearchCV object. Trained model object.
    model_filepath: string. Filepath for where the pickle file should be saved.
    """
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)

def main():
    """
    Main function to load data, build model, train model, evaluate model, and save model.
    """
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the database as the first argument and the filepath of the pickle file to save the model to as the second argument. \n\nExample: python train_classifier.py ../data/AirlineSentiment.db classifier.pkl')

if __name__ == '__main__':
    main()