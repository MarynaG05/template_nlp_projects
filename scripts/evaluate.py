from config_ML import *
import numpy as np
import pandas as pd
import sys
from train import factorize_relation


def naive_bayes_evaluate():
    if (len(sys.argv) < 2):
        print("Please choose 'milestone-1' or 'milestone-2' as an argument")
        sys.exit()

    is_milestone_1 = sys.argv[1] == "milestone-1"
    is_milestone_2 = sys.argv[1] == "milestone-2"

    if not is_milestone_1 and not is_milestone_2:
        print("Please choose 'milestone-1' or 'milestone-2' as an argument")
        sys.exit()

    # Predicting baseline classifier
    if is_milestone_1:
        predictions_filename = naive_bayes_predictions

    # Predicting deep learning classifier
    if is_milestone_2:
        predictions_filename = naive_bayes_predictions # TODO: Change to final deep learning model

    # Reading predictions
    predictions = np.loadtxt(predictions_filename, delimiter=",")

    # Reading test keys
    test_keys = pd.read_csv(test_data_keys_cleaned, header = None)
    # Factorizing test keys
    test_keys['Relation_Number'] = test_keys[0].apply(factorize_relation)
    test_keys_array = np.array(test_keys['Relation_Number'])

    accuracy_for_test_keys = np.mean(predictions == test_keys_array)
    print("Multinomial Naive Bayes Model, Accuracy = {} %".format(accuracy_for_test_keys * 100))

naive_bayes_evaluate()
