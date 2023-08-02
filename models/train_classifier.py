import sys
#import sklearn packages
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import train_test_split
import re
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt')
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
import pickle

def load_data(database_filepath):
    '''
    This function used to load data from ETL database 
    Input: database_filepath
    Output: X, Y, category_names. X = message, Y= categories values and category_name = categories_columns name
    '''
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('etl_output', engine)
    X = df.message
    Y = df.loc[:, df.columns.drop(['id', 'message', 'original', 'genre'])]
    category_names = Y.columns
    return X, Y, category_names

def tokenize(text):
    '''
    This function used to pre-clean data for text before we train our model
    Input : text
    Output : array cleaned_token contains main word that have more meaning in our model
    '''
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")
    
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    return clean_tokens

def build_model():
    '''
    This function used to build model and optimize it using GridSearchCV function
    '''
    pipeline = Pipeline([
    ('vect', CountVectorizer(tokenizer=tokenize)),
    ('tfidf', TfidfTransformer()),
    ('clf', MultiOutputClassifier(AdaBoostClassifier()))
    ])
#     improve model
    parameters = {
        'clf__estimator__learning_rate': [0.5, 1.0],
        'clf__estimator__n_estimators': [10, 20]
    }
    cv = GridSearchCV(pipeline, param_grid=parameters, n_jobs=1, verbose=3)
    return cv

def evaluate_model(model, X_test, Y_test, category_names):
    '''
    This fucntion used to evaluate_model using function classification_report
    '''
    Y_pred = model.predict(X_test)
    for i in range(36):
        print(Y_test.columns[i], ':')
        print(classification_report(Y_test.iloc[:,i], Y_pred[:,i], target_names=category_names))


def save_model(model, model_filepath):
    '''
    This function used to save our model after we trained
    '''
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()