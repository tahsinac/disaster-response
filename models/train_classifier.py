# import packages
# import libraries
import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
import re
import nltk
from nltk.tokenize import word_tokenize,sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.decomposition import TruncatedSVD
import pickle
from workspace_utils import active_session

nltk.download(['punkt', 'wordnet', 'stopwords'])

def load_data(db_filepath):
    """
    Loads data from database and returns X and Y.
    Args:
      db_filepath(str): string filepath of the sqlite database
    Returns:
      X(pandas DataFrame): messages
      Y(pandas DataFrame): labels
    """
    
    # read in file
    engine = create_engine('sqlite:///'+db_filepath)
    df = pd.read_sql_table('msgs_tbl', engine)
    engine.dispose()
    
    # define features and label arrays
    X = df['message']
    Y = df.drop(['id', 'message', 'original', 'genre'], axis = 1)

    return X, Y

def tokenize(text):
    """
    Tokenizes normalized text
    Args:  
        text(string): input text
    Return:
        tokens(list of strings): list of cleaned tokens
    
    """
    #normalizing text: removing punctuations and converting to lowercase
    text  =  re.sub(r'[^a-zA-Z0-9]', ' ', text.lower())
    
    #tokenizing text
    words = word_tokenize(text)
    
    #removing stop words
    words = [w for w in words if w not in stopwords.words('english')]
    
    #lemmatizing
    tokens = [WordNetLemmatizer().lemmatize(w.strip()) for w in words]
    
    return tokens

def build_model():
    """
    Returns a GridSearchCV object to be used as the model for training
    Args:
       None
    Returns:
       cv (GridSearchCV): Grid search model object
    """
    # text processing and model pipeline
    forest_clf = RandomForestClassifier()
    pipeline_2 = Pipeline([
                    ('tfidf', TfidfVectorizer(tokenizer=tokenize)),
                    ('best', TruncatedSVD(n_components=50)),
                    ('clf', MultiOutputClassifier(forest_clf))
                      ])

    # define parameters for GridSearchCV
    parameters= {'clf__estimator__n_estimators': [10, 20], 'clf__estimator__min_samples_split': [100, 120]}

    # create gridsearch object and return as final model pipeline
    cv = GridSearchCV(pipeline_2, param_grid=parameters, n_jobs=-1, cv = 2)

    return cv


def train(X, Y, cv):
    """
     Returns a model trained on training data and prints its classification report
     Args:
        X(pandas DataFrame): messages
        y(pandas DataFrame): labels
     Returns:
        cv : Grid search model object
     """
    # train test split
    xtrain, xtest, ytrain, ytest = train_test_split(X, Y, test_size = 0.20, random_state = 36)
    
    with active_session():
        # fit model
        model = cv.fit(xtrain, ytrain)

        # output model test results
        ypreds = model.best_estimator_.predict(xtest)
        for ind, col in enumerate(ytest):
            print(col)
            print(classification_report(ytest[col], ypreds[:,ind]))
    
    return model


def save_model(model):
    """
    Saves model as pickle file.
    Args:
      cv:  model
    Return:
      N/A
    """
    with open('classifer.pkl', 'wb') as f:
        pickle.dump(model, f)  

def run_pipeline(data_file):
    print('Getting features and labels..')
    X, Y = load_data('DisasterResponse.db')  # run ETL pipeline
    print('Building model..')
    cv = build_model()  # build model pipeline
    print('Training Model..Printing classification report in a few moments..')
    model = train(X, Y, cv)  # train model pipeline
    print('Saving model as pickle..')
    save_model(model)  # save model
    print('Pipeline run successfull!')


if __name__ == '__main__':
    data_file = sys.argv[1]  # get filename of dataset
    run_pipeline(data_file)  # run data pipeline