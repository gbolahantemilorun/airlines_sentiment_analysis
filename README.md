# Airlines Sentiment Analysis

### Table of Contents

1. [Installation](#installation)
2. [Project Overview](#overview)
3. [File Descriptions](#files)
4. [Instructions](#instructions)
5. [Analysis](#analysis)
6. [Conclusion](#conclusion)
7. [Licensing, Authors, and Acknowledgements](#licensing)

## Installation <a name="installation"></a>

To run the Python scripts and web app, make sure to install the following libraries:

#### **process_data.py**
- pandas
- sqlalchemy

#### **run.py**
- flask
- plotly
- nltk

#### **train_classifier.py**
- pandas
- numpy
- sqlalchemy
- nltk
- scikit-learn

## Project Overview <a name="overview"></a>

This project focuses on analyzing sentiment data related to airlines to build a model for an API that classifies the sentiment of airline-related messages. The dataset contains real messages and their associated sentiments (negative, neutral, positive). The goal is to create a machine learning pipeline to categorize these sentiments, enabling effective analysis and response.

The project includes a web app where users can input new messages and receive sentiment classification results. Additionally, the web app displays visualizations of the sentiment distribution used for training the model.

## File Descriptions <a name="files"></a>

### app folder

- **run.py**: This file contains the main script to run the web app.
- **templates folder**: Contains HTML templates for the web app.
    - **go.html**: Template for displaying classification results.
    - **master.html**: Main template for the web app layout.
- **static folder**: Contains CSS file for styling the web app.
    - **style.css**: CSS file for web app styling.

### data folder

- **Tweets.csv**: CSV file containing airline sentiment data.
- **AirlineSentiment.db**: SQLite database file storing cleaned data.
- **process_data.py**: Python script for processing and cleaning data.

### models folder

- **classifier.pkl**: Pickle file containing the trained classifier model.
- **train_classifier.py**: Python script for training the classifier model.

### notebooks folder

- **Airline_Sentiment_ETL_Pipeline**: Notebook for the ETL pipeline.
- **Airline_Sentiment_ML_Pipeline**: Notebook for the ML pipeline.

## Instructions - Running the Python Scripts and Web App <a name="instructions"></a>

1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database:
        `python data/process_data.py data/Tweets.csv data/AirlineSentiment.db`
    - To run ML pipeline that trains classifier and saves the model:
        `python models/train_classifier.py data/AirlineSentiment.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app:
    `python app.py`

3. Go to http://127.0.0.1:3001/

## Analysis <a name="analysis"></a>

1. **Data Cleaning and Processing**:
   - The dataset was loaded and inspected for missing values and duplicates.
   - Text data was tokenized, and stop words were removed.
   - A WordNetLemmatizer was used to lemmatize the tokens.

2. **Model Training**:
   - In the Airline_Sentiment_ML_Pipeline notebook, a machine learning pipeline was created using `CountVectorizer`, `TfidfTransformer`, `RandomForestClassifier`, Gradient Boosting, Multinomial Naive Bayes, and Decision Tree Classifier. However, the random forest model outperformed other models.
   - The model was trained on the cleaned dataset.
   - Cross-validation was used to evaluate the model's performance, resulting in the following metrics:
     - **Accuracy**: 77%
     - **Precision**: 73% (macro avg)
     - **Recall**: 64% (macro avg)
     - **F1-Score**: 67% (macro avg)

3. **Justification for Chosen Metrics**:

    - Given the highly imbalanced nature of the data in this project, it's crucial to choose metrics that adequately address the challenges posed by such an imbalance. Below, I have provided a justification for the selected metrics, specifically tailored to handle the imbalanced dataset in this project.

        - **Precision (per class)**: Precision is crucial to reduce false positives, especially in minority classes like neutral and positive sentiments.

        - **Recall (per class)**: Recall ensures that the model captures most of the relevant instances, reducing false negatives, which is important for comprehensive sentiment detection.

        - **F1-Score (per class and macro average)**: The F1-score, particularly the macro average, balances the performance across all classes, ensuring that the evaluation does not overly favour the majority class (negative). This is essential for an imbalanced dataset to provide a holistic view of the model performance.

        - **Accuracy**: While accuracy gives an overall measure of correctness, it can be misleading in imbalanced datasets. However, it is still useful as a basic metric to gauge the model’s performance. Accuracy indicates a general performance but needs to be supplemented with other metrics for a comprehensive evaluation.

4. **Web App Development**:
   - Due to the Random Forest model's performance, efficiency, scalability and it's robustness to overfitting, it was chosen as the model to be deployed for this project.
   - A Flask web app was developed to allow users to input messages and receive sentiment predictions.
   - The app includes visualizations of sentiment distribution based on the data stored in the database.

## Conclusion <a name="conclusion"></a>

The sentiment analysis model successfully classifies airline-related messages into negative, neutral, and positive sentiments. The web app provides an interactive way to analyze sentiments and can be used by airlines to gain insights into customer feedback. In future work, more model improvement techniques will be explored to improve the performance of the model.

## Licensing, Authors, Acknowledgements <a name="licensing"></a>

This project was developed as part of a data science portfolio project. The dataset was provided by [Kaggle](https://www.kaggle.com/crowdflower/twitter-airline-sentiment). 

Feel free to use the code for educational purposes or adapt it for your own projects, with appropriate credit given to the original authors.
