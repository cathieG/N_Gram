
import re
import os
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
import argparse

def txt_to_df(file_path):
    with open(file_path,'r') as file:
        text = file.read()
        tokens = word_tokenize(text)
        df_tokens = pd.DataFrame(tokens, columns = ['Method Code'])
    return df_tokens

def main():
    parser = argparse.ArgumentParser(description = "Process a text file and convert to DataFrame")
    
    parser.add_argument('file_path', type = str, help = "Path to the txt file")
    
    args = parser.parse_args()

    df = txt_to_df(args.file_path)
    
    print(df.head())

if __name__ == '__main__':
    main()