import re
import os
import pandas as pd
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
import argparse
from pygments.lexers.jvm import JavaLexer
from pygments.lexers import get_lexer_by_name
from pygments.token import Token


def txt_to_df(file_path):
    with open(file_path,'r') as file:
        methods = file.readlines()
        df_methods = pd.DataFrame(methods, columns = ['Method Code'])
    return df_methods

# Removing Type 1 Clones
def remove_duplicates(data):
    """Remove duplicate methods based on method content.
      Almost Type-1 with the exception of comments
    """
    return data.drop_duplicates(subset="Method Code", keep="first")

def filter_ascii_methods(data):
    """Filter methods to include only those with ASCII characters."""
    data = data[data["Method Code"].apply(lambda x: all(ord(char) < 128 for char in x))]
    return data

def remove_outliers(data, lower_percentile=5, upper_percentile=95):
    """Remove outliers based on method length."""
    method_lengths = data["Method Code"].apply(len)
    lower_bound = method_lengths.quantile(lower_percentile / 100)
    upper_bound = method_lengths.quantile(upper_percentile / 100)
    return data[(method_lengths >= lower_bound) & (method_lengths <= upper_bound)]

def remove_boilerplate_methods(data):
    """Remove boilerplate methods like setters and getters."""
    boilerplate_patterns = [
        r"\bset[A-Z][a-zA-Z0-9_]*\(.*\)\s*{",  # Setter methods
        r"\bget[A-Z][a-zA-Z0-9_]*\(.*\)\s*{",  # Getter methods
    ]
    boilerplate_regex = re.compile("|".join(boilerplate_patterns))
    data = data[~data["Method Code"].apply(lambda x: bool(boilerplate_regex.search(x)))]
    return data

def remove_comments(code):
        lexer = get_lexer_by_name('java')
        tokens = lexer.get_tokens(code)
        # Filter out comments using a lambda function
        clean_code = ''.join(token[1] for token in tokens if not (lambda t: t[0] in Token.Comment)(token))

        return clean_code

def remove_comments_from_df(data):
    data['Method Code'] = data['Method Code'].apply(remove_comments)
    return data

def tokenize_methods(df):
    tokenized_methods = []

    for method in df['Method Code']:
        tokens = word_tokenize(method)
        tokenized_methods.append(tokens)
    
    #returning a list of lists
    return tokenized_methods


def main():

    parser = argparse.ArgumentParser(description = "Process a text file and convert to DataFrame")
    parser.add_argument('file_path', type = str, help = "Path to the txt file")
    args = parser.parse_args()

    # convert the txt file to a Dataframe
    df = txt_to_df(args.file_path)
    

    df = remove_duplicates(df)
    print("After removing duplicates:", len(df))

    df = filter_ascii_methods(df)
    print("After removing ASCII methods:", len(df))

    df = remove_outliers(df)
    print("After removing outliers:", len(df))

    df = remove_boilerplate_methods(df)
    print("After removing boilerplate methods:", len(df))

    df = remove_comments_from_df(df)
    print("After removing comments:", len(df))
   
    print("Preprocessing completed.")

    token_list = tokenize_methods(df)
    print(f"Total tokenized methods: {len(token_list)}")


if __name__ == '__main__':
    main()
