import torch
import torch.nn.functional as F
from torch import nn


class BoWClassifierWithEmbedding(nn.Module):
    def __init__(self, num_labels, vocab_size, embedding_dim):
        super(BoWClassifierWithEmbedding, self).__init__()

        # We define the embedding layer here
        # It will convert a list of ids: [1, 50, 64, 2006]
        # Into a list of vectors, one for each word
        # The embedding layer will learn the vectors from the contexts
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=3000)
        # We could also load precomputed embeddings, e.g. GloVe, in some cases we don't want to train the embedding layer
        # In this case we enable the training
        self.embedding.weight.requires_grad = True

        self.linear = nn.Linear(embedding_dim, num_labels)

        #add dropout
        self.dropout = nn.Dropout(0.25)

    def forward(self, text, sequence_lens):
        # First we create the embedded vectors
        embedded = self.embedding(text)
        # We need a pooling to convert a list of embedded words to a sentence vector
        # We could have chosen different pooling, e.g. min, max, average..
        # With LSTM we also do a pooling, just smarter
        pooled = F.max_pool2d(embedded, (embedded.shape[1], 1)).squeeze(1)
        return F.log_softmax(self.linear(pooled), dim=1)

    def save_model(self, filename):
        torch.save(self.state_dict(), filename)

    def load_model(self, filename):
        self.load_state_dict(torch.load(filename))


class LSTMClassifier(nn.Module):
    def __init__(self, num_labels, vocab_size, embedding_dim, hidden_dim):
        super(LSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=3000)
        self.embedding.weight.requires_grad = True

        # Define the LSTM layer
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            batch_first=True,
            num_layers=1,
            bidirectional=False,
        )
        self.linear = nn.Linear(hidden_dim, num_labels)
        # Dropout to overcome overfitting
        self.dropout = nn.Dropout(0.25)

    def forward(self, text, sequence_lens):
        embedded = self.embedding(text)

        # To ensure LSTM doesn't learn gradients for the id of the padding symbol
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, sequence_lens, enforce_sorted=False, batch_first=True
        )
        packed_outputs, (h, c) = self.lstm(packed)
        # extract LSTM outputs (not used here)
        lstm_outputs, lens = nn.utils.rnn.pad_packed_sequence(
            packed_outputs, batch_first=True
        )

        # We use the last hidden vector from LSTM
        y = self.linear(h[-1])
        log_probs = F.log_softmax(y, dim=1)
        return log_probs

    def save_model(self, filename):
        torch.save(self.state_dict(), filename)

    def load_model(self, filename):
        self.load_state_dict(torch.load(filename))