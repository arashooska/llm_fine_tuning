import torch

from mistral.cache import RotatingBufferCache
import logging
import torch
import fire
from typing import List
from pathlib import Path

from mistral.model import Transformer
from mistral.tokenizer import Tokenizer

import pandas as pd

import time

from pathlib import Path

code_path = "/raid/aransari/mistral-src"  # codebase
data_path = Path("/datasets/pruned_data.csv")  # dataset
model_path = Path("/raid/aransari/mistral-7B-v0.1")  # model and tokenizer location

JSON_PATH = "/raid/aransari/datasets/Persian-Wikipedia-Corpus/Json Format of Persian Wikipedia Pages/jsons"

COLS_TO_EXTRACT = "Text"

#Returns a list of dataframes for every json in the directory specified
def load_jsons(json_dir):
    import os
    import pandas as pd

    dataframes = []

    for filename in os.listdir(json_dir):
        if filename.endswith(".json"):

            file_path = os.path.join(json_dir, filename)
            df = pd.read_json(file_path, lines=True)
            dataframes.append(df)


            
    return dataframes

def clean_df(df):

    # Code to extract only the needed rows and columns from the dataframe
    
    #Take only the columns in COLS_TO_EXTRACT and return that sub dataframe
    cleaned_df = df[COLS_TO_EXTRACT]

    return cleaned_df

def main():

    dfs = load_jsons(JSON_PATH)

    print("IN MAIN")

    #Declare an empty dataframe to concatenate all the dataframes into one
    df_list = []

    for i,df in enumerate(dfs):

        #Run some function to clean the dataframe and then attach it to one huge dataframe

        print(f"*************\nThis is DATAFRAME #{i}")

        print(f"DF INFO: {df.info()} \nDF HEAD: {df.head()}")

        df_list.append(clean_df(df))

    df = pd.concat(df_list)

    #Export the DF to CSV file
    df.to_csv("../datasets/persian-wiki-training.csv", index=False)

    



main()
