import csv
import time

import nltk

#nltk.download("punkt")
#nltk.download("wordnet")
#nltk.download("popular")

import pandas as pd
from config_ML import EPOCH_NUMBER
import torch
import warnings
# Set the optimizer and the loss function!
# https://pytorch.org/docs/stable/optim.html
import torch.optim as optim
from nltk import word_tokenize
import numpy as np
from sklearn.exceptions import UndefinedMetricWarning
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split as split
from sklearn.metrics import confusion_matrix, classification_report, multilabel_confusion_matrix
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn import preprocessing
from tuwnlpie.milestone2.model import BoWClassifierWithEmbedding
from torch.nn.utils.rnn import pad_sequence

#from yellowbrick.classifier import ConfusionMatrix


# This is just for measuring training time!
def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def warn(*args, **kwargs):
    pass

class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()

    def __call__(self, articles):
        return [self.wnl.lemmatize(t) for t in word_tokenize(articles)]


class SemevalDataset:
    def __init__(self, train_path, test_keys_path, test_sentences_path, BATCH_SIZE=64):

        # Initialize the correct device
        # It is important that every array should be on the same device or the training won't work
        # A device could be either the cpu or the gpu if it is available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.BATCH_SIZE = BATCH_SIZE

        # Load data
        self.data_unprocessed_train = pd.read_csv(
            train_path, names=["sentence", "label"], header=None
        )
        self.data_unprocessed_test_keys = pd.read_csv(
            test_keys_path, names=["label"], header=None
        )

        self.data_unprocessed_test_sentences = pd.read_csv(
            test_sentences_path, names=["sentence"], header=None
        )

        self.data_unprocessed_test_keys["label"] = self.data_unprocessed_test_keys["label"].str.lstrip('0123456789').str[1:]

        self.data_unprocessed_test = pd.DataFrame()
        self.data_unprocessed_test["sentence"] = self.data_unprocessed_test_sentences
        self.data_unprocessed_test["label"] = self.data_unprocessed_test_keys

        self.tr_data, self.val_data = self.labelEncode(
            self.data_unprocessed_train, self.data_unprocessed_test
        )

        self.word_to_ix = self.prepare_vectorizer(self.data_unprocessed_test)

        (
            self.tr_data_loader,
            self.val_data_loader
        ) = self.prepare_dataloader(
            self.tr_data, self.val_data, self.word_to_ix
        )

        (
            self.train_iterator,
            self.valid_iterator
        ) = self.create_dataloader_iterators(
            self.tr_data_loader,
            self.val_data_loader,
            self.BATCH_SIZE,
        )



        self.an = self.word_to_ix.build_analyzer()

        self.tr_data["length"] = self.tr_data.sentence.str.len()
        self.val_data["length"] = self.val_data.sentence.str.len()
        self.tr_data = self.tr_data.sort_values(by="length")
        self.val_data = self.val_data.sort_values(by="length")

        self.dataset_as_ids = self.create_input(self.tr_data.sentence, self.an, self.word_to_ix.vocabulary_)
        self.padded = pad_sequence(self.dataset_as_ids, batch_first=True, padding_value=3001)

        self.tr_data_loader, self.val_data_loader = self.prepare_dataloader_with_padding(self.tr_data, self.val_data, self.word_to_ix)

        self.train_iterator, self.valid_iterator = self.create_dataloader_iterators_with_padding(
            self.tr_data_loader, self.val_data_loader, BATCH_SIZE
        )

    def getIterators(self):
        return self.train_iterator, self.valid_iterator

    def getXTrain(self):
        return self.data_unprocessed_train["sentence"]

    def getYTrain(self):
        return self.data_unprocessed_train["label"]
    
    def getXTest(self):
        return self.data_unprocessed_test["sentence"]
    
    def getYTest(self):
        return self.data_unprocessed_test["label"]
    
    def read_df_from_csv(self, filename):
        docs = []
        with open(filename) as csvfile:
            reader = csv.reader(csvfile)
            for text, label in tqdm(reader):
                docs.append((text, label))

        df = pd.DataFrame(docs, columns=["text", "label"])

        return df

    def labelEncode(self, data_unprocessed_train, data_unprocessed_test):
        encoder = preprocessing.LabelEncoder()

        data_unprocessed_test["label"] = encoder.fit_transform(
            data_unprocessed_test["label"]
        )
        data_unprocessed_train["label"] = encoder.fit_transform(
            data_unprocessed_train["label"]
        )
        return data_unprocessed_test, data_unprocessed_train

    def split_data(self, train_data, random_seed=2022):
        tr_data, val_data = split(train_data, test_size=0.2, random_state=random_seed)
        tr_data, te_data = split(tr_data, test_size=0.2, random_state=random_seed)

        return tr_data, val_data, te_data

    def prepare_vectorizer(self, tr_data):
        vectorizer = CountVectorizer(
            max_features=3000, tokenizer=LemmaTokenizer(), stop_words="english"
        )

        word_to_ix = vectorizer.fit(tr_data.sentence)

        return word_to_ix

    def create_input(self, dataset, analyzer, vocabulary):
        dataset_as_indices = []

        # We go through each sentence in the dataset
        # We need to add two additional symbols to the vocabulary
        # We have 3000 features, ranged 0-2999
        # We add 3000 as an id for the "unknown" words not among the features
        # 3001 will be the symbol for padding, but about this later!
        for sentence in dataset:
            tokens = analyzer(str(sentence))
            token_ids = []

            for token in tokens:
                # if the token is in the vocab, we add the id
                if token in vocabulary:
                    token_ids.append(vocabulary[token])
                # else we add the id of the unknown token
                else:
                    token_ids.append(3000)

            # if we removed every token during preprocessing (stopword removal, lemmatization), we add the unknown token to the list so it won't be empty
            if not token_ids:
                token_ids.append(3000)
            dataset_as_indices.append(torch.LongTensor(token_ids).to(self.device))

        return dataset_as_indices

    # Preparing the data loaders for the training and the validation sets
    # PyTorch operates on it's own datatype which is very similar to numpy's arrays
    # They are called Torch Tensors: https://pytorch.org/docs/stable/tensors.html
    # They are optimized for training neural networks
    def prepare_dataloader(self, tr_data, val_data, word_to_ix):
        # First we transform the sentences into one-hot encoded vectors
        # Then we create Torch Tensors from the list of the vectors
        # It is also inportant to send the Tensors to the correct device
        # All of the tensors should be on the same device when training
        tr_data_vecs = torch.FloatTensor(word_to_ix.transform(tr_data.sentence).toarray()).to(
            self.device
        )
        tr_labels = torch.LongTensor(tr_data.label.tolist()).to(self.device)

        val_data_vecs = torch.FloatTensor(
            word_to_ix.transform(val_data.sentence).toarray()
        ).to(self.device)
        val_labels = torch.LongTensor(val_data.label.tolist()).to(self.device)

        tr_data_loader = [(sample, label) for sample, label in zip(tr_data_vecs, tr_labels)]
        val_data_loader = [
            (sample, label) for sample, label in zip(val_data_vecs, val_labels)
        ]

        return tr_data_loader, val_data_loader

    # The DataLoader(https://pytorch.org/docs/stable/data.html) class helps us to prepare the training batches
    # It has a lot of useful parameters, one of it is _shuffle_ which will randomize the training dataset in each epoch
    # This can also improve the performance of our model
    def create_dataloader_iterators(self, tr_data_loader, val_data_loader, BATCH_SIZE):
        train_iterator = DataLoader(
            tr_data_loader,
            batch_size=BATCH_SIZE,
            shuffle=True,
        )

        valid_iterator = DataLoader(
            val_data_loader,
            batch_size=BATCH_SIZE,
            shuffle=False,
        )

        return train_iterator, valid_iterator

    def prepare_dataloader_with_padding(self, tr_data, val_data, word_to_ix):
        # First create the id representations of the input vectors
        # Then pad the sequences so all of the input is the same size
        # We padded texts for the whole dataset, this could have been done batch-wise also!
        tr_data_vecs = pad_sequence(
            self.create_input(tr_data.sentence, self.an, word_to_ix.vocabulary_),
            batch_first=True,
            padding_value=3000,
        )
        tr_labels = torch.LongTensor(tr_data.label.tolist()).to(self.device)
        tr_lens = torch.LongTensor(
            [len(i) for i in self.create_input(tr_data.sentence, self.an, word_to_ix.vocabulary_)]
        )

        # We also add the texts to the batches
        # This is for the Transformer models, you wont need this in the next experiments
        tr_sents = tr_data.sentence.tolist()

        val_data_vecs = pad_sequence(
            self.create_input(val_data.sentence, self.an, word_to_ix.vocabulary_),
            batch_first=True,
            padding_value=3000,
        )
        val_labels = torch.LongTensor(val_data.label.tolist()).to(self.device)
        val_lens = torch.LongTensor(
            [len(i) for i in self.create_input(val_data.label, self.an, word_to_ix.vocabulary_)]
        )

        val_sents = val_data.sentence.tolist()

        tr_data_loader = [
            (sample, label, length, sent)
            for sample, label, length, sent in zip(
                tr_data_vecs, tr_labels, tr_lens, tr_sents
            )
        ]
        val_data_loader = [
            (sample, label, length, sent)
            for sample, label, length, sent in zip(
                val_data_vecs, val_labels, val_lens, val_sents
            )
        ]

        return tr_data_loader, val_data_loader

    def create_dataloader_iterators_with_padding(
        self, tr_data_loader, val_data_loader, BATCH_SIZE):
        train_iterator = DataLoader(
            tr_data_loader,
            batch_size=BATCH_SIZE,
            shuffle=True,
        )

        valid_iterator = DataLoader(
            val_data_loader,
            batch_size=BATCH_SIZE,
            shuffle=False,
        )
        
        return train_iterator, valid_iterator

def classifier_model(model, model_features, X_train, X_test, y_train, y_test):
    print(X_train)
    print(X_test)
    X_train = X_train[model_features]
    X_test = X_test[model_features]
    class_model = model
    class_model.fit(X_train, y_train)

    print("Classification Report on Train")
    print(classification_report(y_train, class_model.predict(X_train)))
    print(" ")
    print("Confusion Matrix on Train")
    cm = ConfusionMatrix(class_model, cmap="Blues", fontsize=13)
    cm.score(X_train, y_train)
    cm.show()
    
    print("Classification Report on Test")
    print(classification_report(y_test, class_model.predict(X_test)))
    print(" ")
    print("Confusion Matrix on Test")
    cm = ConfusionMatrix(class_model, cmap="Blues", fontsize=13)
    cm.score(X_test, y_test)
    cm.show()
    
    
    print("Classification Report on Entire Dataset")
    print(classification_report(y_test, class_model.predict(X_test)))
    print(" ")
    print("Confusion Matrix on Entire Dataset")
    cm = ConfusionMatrix(class_model, cmap="Blues", fontsize=13)
    cm.score(X_train.append(X_test), np.append(y_train, y_test))
    cm.show()

    return class_model

class EarlyStopping:
    def __init__(self, tolerance=5, min_delta=0):

        self.tolerance = tolerance
        self.min_delta = min_delta
        self.counter = 0
        self.early_stop = False

    def __call__(self, train_loss, validation_loss):
        if (validation_loss - train_loss) > self.min_delta:
            self.counter +=1
            if self.counter >= self.tolerance:  
                self.early_stop = True

class Trainer:
    def __init__(
        self,
        dataset: SemevalDataset,
        model: BoWClassifierWithEmbedding,
        model_path: str = None,
        test: bool = False,
        lr: float = 0.001,
    ):
        self.dataset = dataset
        self.model = model
        warnings.warn = warn
        warnings.simplefilter("ignore", UserWarning)


        # The optimizer will update the weights of our model based on the loss function
        # This is essential for correct training
        # The _lr_ parameter is the learning rate
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)
        self.criterion = nn.NLLLoss()

        # Copy the model and the loss function to the correct device
        self.model = self.model.to(dataset.device)
        self.criterion = self.criterion.to(dataset.device)

        # Add early_stopping
        self.early_stopping = EarlyStopping(tolerance=5, min_delta=10)

    def calculate_performance(self, preds, y):
        """
        Returns precision, recall, fscore per batch
        """
        # Get the predicted label from the probabilities
        rounded_preds = preds.argmax(1)

        # Calculate the correct predictions batch-wise and calculate precision, recall, and fscore
        # WARNING: Tensors here could be on the GPU, so make sure to copy everything to CPU
        precision, recall, fscore, support = precision_recall_fscore_support(
            rounded_preds.cpu(), y.cpu()
        )

        return precision[1], recall[1], fscore[1]

    def train(self, iterator):

        # We will calculate loss and accuracy epoch-wise based on average batch accuracy
        epoch_loss = 0
        epoch_prec = 0
        epoch_recall = 0
        epoch_fscore = 0

        # You always need to set your model to training mode
        # If you don't set your model to training mode the error won't propagate back to the weights
        self.model.train()

        # We calculate the error on batches so the iterator will return matrices with shape [BATCH_SIZE, VOCAB_SIZE]
        for batch in iterator:
            text_vecs = batch[0]
            labels = batch[1]
            sen_lens = []
            texts = []

            # This is for later!
            if len(batch) > 2:
                sen_lens = batch[2]
                texts = batch[3]

            # We reset the gradients from the last step, so the loss will be calculated correctly (and not added together)
            self.optimizer.zero_grad()

            # This runs the forward function on your model (you don't need to call it directly)
            predictions = self.model.forward(text_vecs, sen_lens)

            # Calculate the loss and the accuracy on the predictions (the predictions are log probabilities, remember!)
            loss = self.criterion(predictions, labels)

            prec, recall, fscore = self.calculate_performance(predictions, labels)

            # Propagate the error back on the model (this means changing the initial weights in your model)
            # Calculate gradients on parameters that requries grad
            loss.backward()
            # Update the parameters
            self.optimizer.step()

            # We add batch-wise loss to the epoch-wise loss
            epoch_loss += loss.item()
            # We also do the same with the scores
            epoch_prec += prec.item()
            epoch_recall += recall.item()
            epoch_fscore += fscore.item()
        return (
            epoch_loss / len(iterator),
            epoch_prec / len(iterator),
            epoch_recall / len(iterator),
            epoch_fscore / len(iterator),
        )
    
    def predict(self, iterator):
        # You always need to set your model to training mode
        # If you don't set your model to training mode the error won't propagate back to the weights
        self.model.train()
        # We calculate the error on batches so the iterator will return matrices with shape [BATCH_SIZE, VOCAB_SIZE]
        test_y = []
        pred_y = []
        for batch in iterator:
            text_vecs = batch[0]
            labels = batch[1]
            sen_lens = []
            texts = []
            #print("LABELS AND SENTENCES\n")
            test_y.append(labels)
            #print(labels)
            #print(text_vecs)
            # This is for later!
            if len(batch) > 2:
                sen_lens = batch[2]
                texts = batch[3]

            # We reset the gradients from the last step, so the loss will be calculated correctly (and not added together)
            self.optimizer.zero_grad()

            # This runs the forward function on your model (you don't need to call it directly)
            predictions = self.model.forward(text_vecs, sen_lens)
            #print("PREDICTIONS\n")  
            pred_y.append(predictions)          
            #print(predictions)
            # Calculate the loss and the accuracy on the predictions (the predictions are log probabilities, remember!)
            loss = self.criterion(predictions, labels)

            prec, recall, fscore = self.calculate_performance(predictions, labels)

            # Propagate the error back on the model (this means changing the initial weights in your model)
            # Calculate gradients on parameters that requries grad
            loss.backward()
            # Update the parameters
            self.optimizer.step()

        fixed_pred = []
        fixed_test = []
        for p in pred_y:
            p = np.argmax(p.detach().numpy(), 1)
            fixed_pred.append(p)
        for p in test_y:
            p = p.detach().numpy()
            fixed_test.append(p)

        #print("PREDICTIONS ARGMAXED")
        #print(fixed_pred)
        #print("ALL LABELS")
        #print(np.concatenate( fixed_test, axis=0 ))
        
        #r = multilabel_confusion_matrix(np.concatenate( fixed_test, axis=0 ), np.concatenate( fixed_pred, axis=0 ), labels=["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"])
        r = confusion_matrix(np.concatenate( fixed_test, axis=0 ), np.concatenate( fixed_pred, axis=0 ))
        np.savetxt("conf_mat.csv", r, delimiter=",")
        df = pd.DataFrame(r, columns=["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"])
        print(df)
        print(dict(zip(['Cause-Effect', 'Component-Whole', 'Content-Container', 'Entity-Destination', 'Entity-Origin', 'Instrument-Agency', 'Member-Collection', 'Message-Topic','Other', 'Product-Producer'], range(10))))
        return np.concatenate( fixed_pred, axis=0 )

    # The evaluation is done on the validation dataset
    def evaluate(self, iterator):

        epoch_loss = 0
        epoch_prec = 0
        epoch_recall = 0
        epoch_fscore = 0
        # On the validation dataset we don't want training so we need to set the model on evaluation mode
        self.model.eval()

        # Also tell Pytorch to not propagate any error backwards in the model or calculate gradients
        # This is needed when you only want to make predictions and use your model in inference mode!
        with torch.no_grad():

            # The remaining part is the same with the difference of not using the optimizer to backpropagation
            for batch in iterator:
                text_vecs = batch[0]
                labels = batch[1]
                sen_lens = []
                texts = []

                if len(batch) > 2:
                    sen_lens = batch[2]
                    texts = batch[3]

                predictions = self.model(text_vecs, sen_lens)
                loss = self.criterion(predictions, labels)

                prec, recall, fscore = self.calculate_performance(predictions, labels)

                epoch_loss += loss.item()
                epoch_prec += prec.item()
                epoch_recall += recall.item()
                epoch_fscore += fscore.item()

        # Return averaged loss on the whole epoch!
        return (
            epoch_loss / len(iterator),
            epoch_prec / len(iterator),
            epoch_recall / len(iterator),
            epoch_fscore / len(iterator),
        )

    def training_loop(self, train_iterator, valid_iterator, epoch_number=EPOCH_NUMBER):
        # Set an EPOCH number!
        N_EPOCHS = epoch_number

        best_valid_loss = float("inf")

        # We loop forward on the epoch number
        for epoch in range(N_EPOCHS):

            start_time = time.time()

            # Train the model on the training set using the dataloader
            train_loss, train_prec, train_rec, train_fscore = self.train(train_iterator)
            # And validate your model on the validation set
            valid_loss, valid_prec, valid_rec, valid_fscore = self.evaluate(valid_iterator)

            # early stopping
            self.early_stopping(train_loss, valid_loss)
            if self.early_stopping.early_stop:
                print("We are at epoch:", epoch)
                break
            end_time = time.time()

            epoch_mins, epoch_secs = epoch_time(start_time, end_time)

            # If we find a better model, we save the weights so later we may want to reload it
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss

            print(f"Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s")
            print(
                f"\tTrain Loss: {train_loss:.3f} | Train Prec: {train_prec*100:.2f}% | Train Rec: {train_rec*100:.2f}% | Train Fscore: {train_fscore*100:.2f}%"
            )
            print(
                f"\t Val. Loss: {valid_loss:.3f} |  Val Prec: {valid_prec*100:.2f}% | Val Rec: {valid_rec*100:.2f}% | Val Fscore: {valid_fscore*100:.2f}%"
            )

        return best_valid_loss
