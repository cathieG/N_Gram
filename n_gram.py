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
#from Lab1_Ngram:
from nltk.lm import MLE
from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk import ngrams
from nltk import FreqDist
from nltk.lm import Laplace
from nltk.lm.models import NgramModel
import numpy as np
import matplotlib.pyplot as plt
import collections
import ast
import random
import concurrent.futures
import heapq
import math
import json
import csv

def txt_to_df(file_path):
    with open(file_path,'r') as file:
        methods = file.readlines()
        df_methods = pd.DataFrame(methods, columns = ['Method Code'])
    return df_methods

def txt_to_list_of_lists(filename):
    with open(filename, "r") as file:
        return [ast.literal_eval(line.strip()) for line in file]


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


def build_ngram_tokens_list(tokens_list, n):
  ngrams = []
  for items in tokens_list:
    ngrams.extend([tuple(items[i:i+n]) for i in range(len(items) - n + 1)])
  return ngrams


def eval_ngram_model_nltk(eval_data, n, lm):
  print(n)
  test_eval_data = build_ngram_tokens_list(eval_data, n)

  return lm.perplexity(test_eval_data)


def generate_next_token(context):
  next_token = lm.generate(1, context)
  prob = lm.score(next_token, context)
  return next_token, prob


def check_balance(tokens):
  stack = ["."]
  matching_bracket = {"}": "{"}

  for token in tokens:
    if token == "{":
      if stack[0] == ".":
        stack.pop()
      stack.append(token)
    elif token == "}":
      if not stack or stack[-1] != matching_bracket[token]:
        return False
      stack.pop()

  return len(stack) == 0


def chain_generate_tokens(starting_context, context_size, max_length):
    generated_tokens = starting_context
    output_pair = []


    for i in range(max_length):
      context = generated_tokens[-(context_size):]
      next_token, prob = generate_next_token(context)

      if next_token == None:
        break

      generated_tokens.append(next_token)
      output_pair.append((next_token, prob))

      if check_balance(generated_tokens):
        break

    ngram_list = list(ngrams(generated_tokens, context_size + 1))
    perp = lm.perplexity(ngram_list)

    return output_pair, perp

def test_ngram_model(n, testing_list, filename, lm):

  count = len(testing_list)
  total_perplexity = 0

  with open(filename, 'w') as json_file:

    for i in range(len(testing_list)):
      starting_context = testing_list[i][:n-1]
      output_list, perp = chain_generate_tokens(starting_context, n-1, 512)

      total_perplexity += perp
      output_string = "[" + str(i+1) + "]: "  + str(output_list)
      # print(output_string)

      if i < 100:
        json.dump(output_string, json_file)
        json_file.write('\n')
    
      if i%50 == 0: # print statement for debugging
        print(f"-----------------{i}------------------")

  print("Model perplexity is: "+ str(total_perplexity/count))

def process_ngram_model_teacher(teacher_token_list, token_list, filename):

  random.shuffle(token_list)

  evaluation_data = token_list[:8000]
  testing_data = token_list[8000:16000]
  #training_data = token_list[16000:80000]

  print("Starting training process:")
  print()
  best_perplexity = []
  for n in [3, 5, 7]:
    train_data, vocab = padded_everygram_pipeline(n, teacher_token_list)

    lm = Laplace(n)
    lm.fit(train_data, vocab)

    perp = eval_ngram_model_nltk(evaluation_data, n, lm)

    print(n)
    print(perp)
    print()

    if len(best_perplexity) == 0:
      best_perplexity.append((n,perp))
      best_lm = lm

    elif len(best_perplexity) != 0:
      if best_perplexity[0][1] > perp:
        best_perplexity[0] = (n, perp)
        best_lm = lm

  best_n = best_perplexity[0][0]
  print("Best n is: " + str(best_n))
  print()

  test_ngram_model(best_n, testing_data, filename, best_lm)

def main():

    parser = argparse.ArgumentParser(description = "Process a text file and convert to DataFrame")
    parser.add_argument('file_path', type = str, help = "Path to the txt file")
    args = parser.parse_args()

    # convert the user txt file to a Dataframe
    df = txt_to_df(args.file_path)
    
    # convert our txt file to a list of lists
    our_data = txt_to_list_of_lists("mydata.txt")

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

    #training, testing, and output
    filename = 'results_teacher_model.json'

    process_ngram_model_teacher(token_list, our_data, filename)


if __name__ == '__main__':
    main()