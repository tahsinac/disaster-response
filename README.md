# Disaster Response Pipeline Project
### Project Overview
In this project, disaster data from Figure Eight is analyzed to build a model for an API that classifies disaster messages.

The data set contains real messages that were sent during disaster events. A machine learning pipeline  is created to categorize these events so that in the event of a disaster, one can send the messages to an appropriate disaster relief agency.

The project includes a web app where an emergency worker can input a new message and get classification results in several categories. The web app will also display visualizations of the data.

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

### Required packages:
- flask
- joblib
- pandas
- plot.ly
- numpy
- scikit-learn
- sqlalchemy

### Files In This Repo
- classifer.pkl : trained model saved as pickel file from the ML pipeline
- DisasterResponse.db : SQL database containing cleaned data, produced by the ELT pipeline
- train_classifier.py : python script to train model
- process_data.py : python script to load data from csv files, clean it and save as SQL database.
