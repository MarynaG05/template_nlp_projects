from config_ML import *
import pandas as pd
import numpy as np
import pickle
import sys
from train import EMBEDDING_DIM, HIDDEN_DIM, INPUT_DIM, OUTPUT_DIM
from tuwnlpie.milestone2.model import BoWClassifierWithEmbedding, LSTMClassifier

from tuwnlpie.milestone2.utils import SemevalDataset, Trainer

def predict():
    if (len(sys.argv) < 2):
        print("Please choose 'milestone-1' or 'milestone-2' as an argument")
        sys.exit()

    is_milestone_1 = sys.argv[1] == "milestone-1"
    is_milestone_2 = sys.argv[1] == "milestone-2"
    if (is_milestone_2 and (len(sys.argv) != 3)):
        print("Please choose a classifier with: 'BoW' or 'LSTM' as the second argument")
        sys.exit()
    elif is_milestone_2:
        model_choice = sys.argv[2]

    if not is_milestone_1 and not is_milestone_2:
        print("Please choose 'milestone-1' or 'milestone-2' as an argument")
        sys.exit()

    # Predicting baseline classifier
    if is_milestone_1:
        model_filename = './tuwnlpie/milestone1/baseline_classifier'
        predictions_filename = naive_bayes_predictions
        # Loading model
        model = pickle.load(open(model_filename, 'rb'))

        # Reading cleaned test data (sentences)
        test_col_names = ['Sentence']
        test_data = pd.read_csv(test_data_sentences_cleaned, header = None, names = test_col_names)
        test_data = pd.DataFrame(test_data)
        test_data_str = test_data['Sentence'].astype(str)

        # Making predictions
        predictions = model.predict(test_data_str)


    # Predicting deep learning classifier
    if is_milestone_2:
        if model_choice != "LSTM":
            filename = './tuwnlpie/milestone2/BoWClassifierWithEmbedding'
            predictions_filename = bow_classifier_predictions
            model = BoWClassifierWithEmbedding(OUTPUT_DIM, INPUT_DIM, EMBEDDING_DIM)
        else: 
            filename = './tuwnlpie/milestone2/LSTMClassifier'
            predictions_filename = lstm_classifier_predictions
            model = LSTMClassifier(OUTPUT_DIM, INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM)

        dataset = SemevalDataset(training_data_cleaned, test_data_keys, test_data_sentences_cleaned)
        _, valid_iterator = dataset.getIterators()
        model.load_model(filename)
        trainer = Trainer(dataset, model) 
        predictions = trainer.predict(valid_iterator)   



    # Save Predictions to csv file
    np.savetxt(predictions_filename, predictions, delimiter=",")
    print('Predictions saved to a csv file succesfully')

predict()
