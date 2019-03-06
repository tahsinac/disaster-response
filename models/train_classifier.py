# import packages
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

def load_data(db_filepath, tbl_name):
    """
    Loads data from database and returns X and Y.
    Args:
      db_filepath(str): string filepath of the sqlite database
      tbl_name:(str): table name in the database file.
    Returns:
      X(pandas DataFrame): messages
      Y(pandas DataFrame): labels
    """
    
    # read in file
    engine = create_engine('sqlite:///{}'.format(data_filepath))
    df = pd.read_sql_table(tbl_name, engine)
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
    pipeline_2 = Pipeline([
                    ('tfidf', TfidfVectorizer(tokenizer=tokenize)),
                    ('best', TruncatedSVD(n_components=100)),
                    ('clf', MultiOutputClassifier(forest_clf))
                      ])

    # define parameters for GridSearchCV
    param_grid = {
        'tfidf__ngram_range': ((1, 1), (1, 2)),
        'tfidf__max_df': (0.8, 1.0),
        'tfidf__max_features': (None, 10000),
        'best__n_components': (10, 50, 100),
        'clf__estimator__n_estimators': [10, 20, 50, 100],
        'clf__estimator__min_samples_split': [20, 40]
    }

    # create gridsearch object and return as final model pipeline
    cv = GridSearchCV(pipeline_2, param_grid=parameters, n_jobs=-2)

    return cv


def train(X, y, model):
    """
     Returns a model trained on training data and prints its classification report
     Args:
        X(pandas DataFrame): messages
        y(pandas DataFrame): labels
     Returns:
        model : 
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


def export_model(model):
    # Export model as a pickle file
    with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)


def run_pipeline(data_file):
    X, y = load_data(data_file)  # run ETL pipeline
    model = build_model()  # build model pipeline
    model = train(X, y, model)  # train model pipeline
    export_model(model)  # save model


if __name__ == '__main__':
    data_file = sys.argv[1]  # get filename of dataset
    run_pipeline(data_file)  # run data pipeline