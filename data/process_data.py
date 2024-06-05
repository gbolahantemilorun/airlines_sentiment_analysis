import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(filepath):
    """
    Load the dataset from a CSV file.

    Args:
    filepath (str): Path to the CSV file.

    Returns:
    pd.DataFrame: Loaded data as a pandas DataFrame.
    """
    return pd.read_csv(filepath)

def clean_data(df):
    """
    Clean the data by removing duplicates, handling missing values,
    converting columns to appropriate data types, and encoding the target variable.

    Args:
    df (pd.DataFrame): Input DataFrame.

    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    # Remove duplicates
    df = df.drop_duplicates()

    # Drop irrelevant columns
    df.drop(['tweet_id', 'airline_sentiment_gold', 'negativereason_gold'], axis=1, inplace=True)

    # Handle missing values
    df['negativereason'] = df.apply(lambda row: 'Unknown' if row['airline_sentiment'] == 'negative' and pd.isna(row['negativereason']) else row['negativereason'], axis=1)
    df['negativereason_confidence'].fillna(0, inplace=True) # Fill missing values with 0
    df['tweet_location'].fillna('Unknown', inplace=True) # Fill missing values with Unknown

    # Convert 'tweet_created' to datetime
    df['tweet_created'] = pd.to_datetime(df['tweet_created'], errors='coerce')

    # Clean the text column
    df['text'] = df['text'].str.replace(r'@\w+', '', regex=True).str.replace(r'http\S+', '', regex=True).str.strip()

    # Encode the target variable
    sentiment_mapping = {'negative': 0, 'neutral': 1, 'positive': 2}
    df['airline_sentiment'] = df['airline_sentiment'].map(sentiment_mapping)

    return df

def save_data(df, database_filepath, table_name):
    """
    Save the cleaned DataFrame to a SQLite database.

    Args:
    df (pd.DataFrame): Input DataFrame.
    database_filepath (str): Filepath for the SQLite database.
    table_name (str): Name of the table to save the data in.
    """
    engine = create_engine(f'sqlite:///{database_filepath}')
    df.to_sql(table_name, engine, index=False, if_exists='replace')

def main():
    """
    Main function to load, clean, and save data.
    """
    if len(sys.argv) == 3:
        filepath, database_filepath = sys.argv[1:]

        print(f'Loading data...\n    FILE: {filepath}')
        df = load_data(filepath)

        print('Cleaning data...')
        df = clean_data(df)

        print(f'Saving data...\n    DATABASE: {database_filepath}')
        save_data(df, database_filepath, 'tweets')

        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepath of the dataset as the first argument '
              'and the filepath of the database to save the cleaned data to as the second argument. \n\n'
              'Example: python etl_pipeline.py Tweets.csv AirlineSentiment.db')

if __name__ == "__main__":
    main()