import os
import torch

this_dir = os.path.dirname(os.path.realpath(__file__))
this_dir = os.path.dirname(os.path.realpath(__file__))

RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
torch.backends.cudnn.deterministic = True

# Input File Paths
training_data = os.path.join(this_dir, '../data/training/TRAIN_FILE.TXT')
test_data_sentences = os.path.join(this_dir, '../data/test/TEST_FILE_CLEAN.TXT')
test_data_keys = os.path.join(this_dir, '../data/test/TEST_FILE_KEY.TXT')

# Output File Paths
training_data_cleaned = os.path.join(this_dir, '../data/training/train_data.csv')
test_data_sentences_cleaned = os.path.join(this_dir, '../data/test/test_data_sentences.csv')
test_data_keys_cleaned = os.path.join(this_dir, '../data/test/test_data_keys.csv')
naive_bayes_predictions = os.path.join(this_dir, '../data/predictions/naive_bayes_predictions.csv')
bow_classifier_predictions = os.path.join(this_dir, '../data/predictions/bow_classifier_predictions.csv')
lstm_classifier_predictions = os.path.join(this_dir, '../data/predictions/lstm_classifier_predictions.csv')

#Neural network config
EPOCH_NUMBER = 20
#BoWClassifierWithEmbedding
EMBEDDING_DIM_BOW = 500

#LSTMClassifier
EMBEDDING_DIM_LSTM = 100
HIDDEN_DIM_LSTM = 20

#Semeval dataset
VOCAB_SIZE_SEMEVAL = 3000
OUTPUT_DIM_SEMEVAL = 10