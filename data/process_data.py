import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    
    """loads the specified message and category data
    Args:
        messages_filepath (string): The file path of the messages.csv
        categories_filepath (string): The file path of the categories.csv
    Returns:
        df: pandas dataframe containing messages and categories 
    """
    
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, on = 'id')
    return df

def clean_data(df):
    
    """Cleans the data:
        - drops duplicates
        - cleans up the categories column
    Args:
        df: pandas dataframe containing messages and categories 
    Returns:
        df:  cleaned pandas dataframe with messages and split categories
    """
    
    categories = df['categories'].str.split(';', expand = True)
    row = categories.loc[0,:]
    category_colnames = row.apply(lambda x:x.split('-')[0]).values.tolist()
    categories.columns = category_colnames
    for column in categories:
        categories[column] = categories[column].apply(lambda x: x.split('-')[1])
        categories[column] = categories[column].astype(int)
    df.drop('categories',axis = 1, inplace = True)
    df = pd.concat([df, categories], axis = 1)
    df.drop_duplicates(subset = 'message', inplace = True)
    return df

def save_data(df, database_filename):
    
    """Saves the resulting data to a sqlite db
    Args:
        df:  cleaned pandas dataframe with messages and split categories
        database_filename (string): the file path to save the db
    Returns:
        None
    """
    
    engine = create_engine('sqlite:///{}.db'.format(database_filename))
    df.to_sql(database_filename, engine, index=False)  


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()