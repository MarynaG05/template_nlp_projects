# Imports
import sys
import os
import re
import csv
import pandas as pd
import numpy as np
import random
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV
from config_ML import *
from tuwnlpie.milestone2.model import BoWClassifierWithEmbedding, LSTMClassifier
from tuwnlpie.milestone2.utils import SemevalDataset, Trainer, classifier_model


if (len(sys.argv) < 2):
    print("Please choose 'milestone-1' or 'milestone-2' as an argument")
    sys.exit()

is_milestone_1 = sys.argv[1] == "milestone-1"
is_milestone_2 = sys.argv[1] == "milestone-2"

if (is_milestone_2 and (len(sys.argv) != 3)):
    print( "Please choose a classifier with: 'BoW' or 'LSTM' as the second argument")
    sys.exit()
elif is_milestone_2:
    model_choice = sys.argv[2]
    


if (not is_milestone_1) and (not is_milestone_2):
    print("Please choose 'milestone-1' or 'milestone-2' as an argument")
    sys.exit()

def read_data_by_line(input_data):
    all_lines = []
    with open(input_data, 'r') as file:
        temp_lines = file.readlines()
        for line in temp_lines:
            line = line.strip()
            if line:
                all_lines.append(line)
    return all_lines

def load_data(input_data, is_training_data):
    data_lines = read_data_by_line(input_data)
    data_raw = []
    i = 0
    for line in data_lines:
        # remove 'Comment' line for training data
        if is_training_data:
            if not line.startswith('Comment:'):
                data_raw.append(line.split('|'))
        else:
            data_raw.append(line.split('|'))
        i += 1

    sentences = []
    relations = []

    for i,j in enumerate(data_raw, 1):
        if is_training_data:
            if i%2 == 0:
                relations.append(j)
            else:
                sentences.append(j)
        else:
            sentences.append(j)

    e1e2 = []
    for sentence_line in sentences:
        sentence_string = ''.join(sentence_line)
        sentence_match = re.sub('\t','',sentence_string)
        k=sentence_match.lstrip('0123456789.-')
        sentence_sub = re.sub('<[^>]*>', '', k)
        sentence_line_final = sentence_sub.replace('"', '')
        e1e2.append(sentence_line_final)

    # converting to strings
    relations_string = ','.join(str(r) for v in relations for r in v)
    w2 = re.sub(r'\([^)]*\)', "",relations_string).split(',')

    # Generating sentence-relation tuples for the csv
    data_output = zip(e1e2,w2)

    if is_training_data:
        with open(training_data_cleaned, "w") as train_file:
            writer = csv.writer(train_file,delimiter=',')
            for row in data_output:
                writer.writerow(row)
    else:
        with open(test_data_sentences_cleaned, "w") as test_sentences_file:
            writer = csv.writer(test_sentences_file)
            for row in e1e2:
                writer.writerow([row])

def load_keys_data(input_data):
        with open(input_data, 'r') as file:
            key_value = []
            key_lines = file.readlines()

            for line in key_lines:
                key_line = line.strip()
                key_string = ''.join(key_line)
                key_match = re.sub('\t \r \n','',key_string)
                key_rem_num = key_match.lstrip('0123456789.-')
                key_rem_num = re.sub('\t','',key_rem_num)
                key_value.append(key_rem_num)

            with open(test_data_keys_cleaned, "w") as test_keys_file:
                writer_key = csv.writer(test_keys_file,delimiter=',')
                for row_key in key_value:
                    writer_key.writerow([row_key])

def factorize_relation(x):
    if x=='Other':
        return 0
    if x=='Cause-Effect':
        return 1
    if x=='Product-Producer':
        return 2
    if x=='Entity-Origin':
        return 3
    if x=='Instrument-Agency':
        return 4
    if x=='Component-Whole':
        return 5
    if x=='Content-Container':
        return 6
    if x=='Entity-Destination':
        return 7
    if x=='Member-Collection':
        return 8
    if x=='Message-Topic':
        return 9

# Loading, cleaning and saving cleaned data
load_data(training_data, True)
load_data(test_data_sentences, False)
load_keys_data(test_data_keys)

# Reading cleaned training data
train_col_names = ['Sentence','Relation']
train_data = pd.read_csv(training_data_cleaned, header = None, names = train_col_names)
train_data = pd.DataFrame(train_data)
train_data_str = train_data['Sentence'].astype(str)
train_data_relation = train_data[['Relation']]

# Reading cleaned test data (sentences)
test_col_names = ['Sentence']
test_data = pd.read_csv(test_data_sentences_cleaned, header = None, names = test_col_names)
test_data = pd.DataFrame(test_data)
test_data_str = test_data['Sentence'].astype(str)

# Reading cleaned test data (keys)
test_keys = pd.read_csv(test_data_keys_cleaned, header = None)

# Factorizing train relations
train_data_relation['Relation_Number'] = train_data_relation['Relation'].apply(factorize_relation)
train_data_relation_array = np.array(train_data_relation['Relation_Number'])

# Factorizing test keys
test_keys['Relation_Number'] = test_keys[0].apply(factorize_relation)
test_keys_array = np.array(test_keys['Relation_Number'])

# Building Baseline Model: Multinomial Naive naive_bayes
if is_milestone_1:
    classifier = MultinomialNB(alpha=0.01)
    filename = './tuwnlpie/milestone1/baseline_classifier'
    # Preprocessing data using pipeline and including classifier as final step
    text_classification = Pipeline([('vectorizer', CountVectorizer(ngram_range=(1,3))), ('tfidftransformer', TfidfTransformer(use_idf=True)), ('classification', classifier)])

    # Training model
    text_classification = text_classification.fit(train_data_str, train_data_relation_array)

    # Saving trained model
    pickle.dump(text_classification, open(filename, 'wb'))
    print("Model trained succesfully.")

# Building Deep Learning Model: BoWClassifierWithEmbedding or LSTMClassifier
if is_milestone_2:
    INPUT_DIM = VOCAB_SIZE_SEMEVAL + 2
    OUTPUT_DIM = OUTPUT_DIM_SEMEVAL
    EMBEDDING_DIM = EMBEDDING_DIM_BOW
    HIDDEN_DIM = HIDDEN_DIM_LSTM

    if model_choice != "LSTM":
        model = BoWClassifierWithEmbedding(OUTPUT_DIM, INPUT_DIM, EMBEDDING_DIM)
    else:  
        model = LSTMClassifier(OUTPUT_DIM, INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM)
    

    dataset = SemevalDataset(training_data_cleaned, test_data_keys, test_data_sentences_cleaned)
    train_iterator, valid_iterator = dataset.getIterators()
    trainer = Trainer(dataset, model) 
    trainer.training_loop(train_iterator, valid_iterator)
    
    if model_choice != "LSTM":
        filename = './tuwnlpie/milestone2/BoWClassifierWithEmbedding'
    else:  
        filename = './tuwnlpie/milestone2/LSTMClassifier'

    model.save_model(filename)
    print("Model trained succesfully.")


# Make predictions
# predictions = naive_bayes_predict(text_classification, test_data_str)

# Evaluate predictions
# naive_bayes_evaluate(predictions, test_keys_array)
