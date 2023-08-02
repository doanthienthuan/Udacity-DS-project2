import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''
    This function will be used to load data from 2 csv files in the same   folder
    Input: 
    1/ message_filepath (filepath of disater_messages.csv)
    2/ categories_filepath (filepath of disater_categories.csv
    Output:
    A dataframe that contain 2 dataframe merged by id
    '''
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    #merge data
    df = categories.merge(messages, left_on='id', right_on='id', how='inner')
    return df


def clean_data(df):
    '''
    This function will be used to clean data. What we will do in this function are:
    1/ create a dataframe name categories using columns named "categories" from input and split by ';'. Because in categories columns, each of category is seperated by ; and we want each of category will a column in our dataframe. 
    2/ After that we apply some functions to changes data type and merge it back to the main dataframe. 
    3/Final we will clean it by drop some duplicated rows.
    Input: df => the merged dataframe
    Output: the dataframe after we cleaned (expected to have 36 categories columns with 1 = True and 0 = False)
    '''
    # create a dataframe of the 36 individual category columns
    categories = df["categories"].str.split(';', expand=True)
    # select the first row of the categories dataframe
    row = categories[0:1]
    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    category_colnames = row.apply(lambda x: x.str[:-2]).values.tolist()
    categories.columns = category_colnames
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].astype('str').str[-1]

        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])
    df.drop(['categories'], axis=1, inplace = True)
    df = pd.concat([df, categories], axis=1)
    final_df = df.drop_duplicates()
    return final_df

def save_data(df, database_filename):
    '''
    This function used to create database and write dataframe result to a table
    Input: df and database_filename
    Output: We don't return anything but we expect to have a .db file in the same folder
    '''
    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql('etl_output', engine, if_exists = 'replace', index=False)


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