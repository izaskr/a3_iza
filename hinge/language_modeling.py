import argparse
import time
import math
import os
import torch
import torch.nn as nn
import torch.onnx
import torch.optim as optim
import torch.nn.functional as F
from glove import load_glove_embeddings, get_glove_embeddings
import json
import data
import model_forwardweights
import model # nn where forward returns softmaxed output
from hinge_loss import hinge, hinge_max, hinge_dist
import itertools
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

class EventLanguageModeling():
    """
    Class that includes all event LMs; runs the training and evaluation; returns the loss and logprobs
    """

    def __init__(self, id2map, list_eventIDs, corpus, batch_size, seq_len, bptt, model_type, ntokens, emsize, nhid, nlayers, glove_vocab, padidx, dropout, tied, device, lr, epochs, anneal, hinge_type):
        #     model = model.RNNModel(args.model, ntokens, args.emsize, args.nhid, args.nlayers,
        #     glove_embeddings_vocab, corpus.pad_idx, args.dropout, args.tied).to(device)
        self.id2map = id2map
        self.list_eventIDs = list_eventIDs
        self.model_type = model_type
        self.ntokens = ntokens
        self.emsize = emsize
        self.nhid = nhid
        self.nlayers = nlayers
        self.glove_embeddings_vocab = glove_vocab
        self.pad_idx = padidx
        self.dropout = dropout
        self.tied = tied
        self.device = device
        self.lr = lr
        self.epochs = epochs
        self.anneal = anneal
        self.hinge_type = hinge_type

        self.nn_models = []
        self.initialize_nn_models()

        # the corpus arg is an instance of the Corpus class from data.py
        # corpus has the following attributes: self.train_sents, self.train_events,
        # self.valid_sents, self.valid_events, self.test_sents, self.test_events
        # self.id2event, self.dictionary (vocab)
        # pad_token, bos_token, eos_token
        # the method iter_batches

        self.batch_size = batch_size
        self.seq_len = seq_len
        self.bptt = bptt

        self.epoch_losses = []
        self.train_losses, self.val_losses = [], []
        self.training_epochs(corpus)
        print("Train and val losses ", self.train_losses, "\n", self.val_losses)

        self.plot_curves()

    def initialize_nn_models(self):
        for eventID in self.list_eventIDs:
            #print(eventID)
            self.nn_models.append(model_forwardweights.RNNModel(self.model_type, self.ntokens, self.emsize, self.nhid, self.nlayers, self.glove_embeddings_vocab, self.pad_idx, self.dropout, self.tied).to(self.device))

        print("Number of created NN models %d , number of events %d should be equal" % (len(self.nn_models), len(self.list_eventIDs)))

        # initialize the optimizer and the LR scheduler
        #all_params = itertools.chain.from_iterable([model.parameters() for model in self.nn_models])
        #self.optimizer = optim.Adam(all_params, self.lr)
        #self.scheduler = optim.lr_scheduler.StepLR(optimizer=self.optimizer, step_size=1, gamma=0.1)

    def repackage_hidden(self, h):
        # Wraps hidden states in new Tensors, to detach them from their history.
        if isinstance(h, torch.Tensor):
            return h.detach()
        else:
            return tuple(self.repackage_hidden(v) for v in h)

    def train_step(self, corpus):
        """
        Performs one train step (one epoch): forward pass and backward pass
        """

        all_params = itertools.chain.from_iterable([model.parameters() for model in self.nn_models])
        self.optimizer = optim.Adam(all_params, self.lr)

        for model in self.nn_models:
            model.train()

        total_loss = 0.

        ### OLD: data first
        batch_counter = 0
        for batch, (data, targets, events) in enumerate(corpus.iter_batches(
                corpus.train_sents, corpus.train_events, batch_size=self.batch_size,
                seq_len=self.bptt, bptt=self.bptt, device=self.device)):

            batch_counter += 1

            # data shape is seq_len × batch_size
            batch_emission_matrix = torch.zeros((len(self.list_eventIDs), data.shape[1]), device=self.device)
            # actually I should store the logprobs across all batches, so at the end of batch looping
            # the emission_matrix should be filled

            for eventID, model in enumerate(self.nn_models):
                # initialize hidden state of LSTM
                if self.model_type != "Transformer":
                    hidden = model.init_hidden(self.batch_size)

                
                self.optimizer.zero_grad()

                # forward pass
                if self.model_type == 'Transformer':
                    output = model(data)
                    output = output.view(-1, self.ntokens)
                else:
                    hidden = self.repackage_hidden(hidden)
                    output, hidden = model(data, hidden)

                #print("softmax output shape", output.shape)

                softmaxed_output = output.reshape(self.seq_len-1, self.batch_size, self.ntokens)
                #print("softmax output reshaped", softmaxed_output.shape)
                #print("batch emission matrix shape", batch_emission_matrix.shape)
                # reshape targets into the shape seq_len-1 × batch_size to iterate over words and accumulate the
                # log prob of a segment. one column is 1 segment, skipping the <bos> and
                # ignoring <pad> symbol!
                batch_segments = targets.reshape(data.shape)
                for word_index, words in enumerate(batch_segments):
                    probs = []
                    for column, word in enumerate(words):
                        probs.append(softmaxed_output[word_index][column][word])
                    for j, logvalue in enumerate(probs):
                        if words[j] != self.pad_idx:
                            batch_emission_matrix[eventID][j] += logvalue
                        #else:
                        #    print("-----------------Pad token",words[j], self.pad_idx)

            #import pdb; pdb.set_trace()


            if self.hinge_type == "max":
                batch_loss = hinge_max(matrix_emissions=batch_emission_matrix, gold_events=events, delta=1, device=self.device, mode="train")

            if self.hinge_type == "dist":
                batch_loss = hinge_dist(matrix_emissions=batch_emission_matrix, gold_events=events, delta=1,
                                       device=self.device, mode="train")


            batch_loss.backward()
            # gradient clipping
            for m, mod in enumerate(self.nn_models):
                # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
                torch.nn.utils.clip_grad_norm_(mod.parameters(), 0.5)

            # note that the weight update is done after the forward passes for the current batch for all events
            self.optimizer.step()

            #for model in self.nn_models:
            #    model.zero_grad()

            self.optimizer.zero_grad()

            total_loss += batch_loss.item()
        

        # average of the total loss with the number of batches
        total_loss = total_loss / batch_counter
        print("\t TOTAL EPOCH LOSS", total_loss)

        return total_loss

    def evaluate(self, data_source):
        """ data source is an iterator over the train or val data """

        # iterate over the all models; to each model pass the entire dataset and get the loss
        # turn on eval mode to disable dropout
        total_loss = 0.

        pass_counter = 0

        with torch.no_grad():

            for eventID, model in enumerate(self.nn_models):
                model.eval()
                i = 0 # batch counter
                model_loss = 0.

                if self.model_type != "Transformer":
                    hidden = model.init_hidden(self.batch_size)

                for no_batch, (data, targets, events) in enumerate(data_source):
                    i += 1
                    pass_counter += 1
                    batch_emission_matrix = torch.zeros((len(self.list_eventIDs), data.shape[1]), device=self.device)

                    # forward pass
                    if self.model_type == 'Transformer':
                        output = model(data)
                        output = output.view(-1, self.ntokens)
                    else:
                        hidden = self.repackage_hidden(hidden)
                        output, hidden = model(data, hidden)


                    softmaxed_output = output.reshape(self.seq_len - 1, self.batch_size, self.ntokens)

                    # reshape targets into the shape seq_len-1 × batch_size to iterate over words and accumulate the
                    # log prob of a segment. one column is 1 segment, skipping the <bos> and
                    # ignoring <pad> symbol!
                    batch_segments = targets.reshape(data.shape)
                    for word_index, words in enumerate(batch_segments):
                        probs = []
                        for column, word in enumerate(words):
                            probs.append(softmaxed_output[word_index][column][word])
                        for j, logvalue in enumerate(probs):
                            if words[j] != self.pad_idx:
                                batch_emission_matrix[eventID][j] += logvalue


                    #batch_loss = hinge(batch_emission_matrix, events, 5, self.device, mode="evaluate")
                    if self.hinge_type == "max":
                        batch_loss = hinge_max(batch_emission_matrix, events, 5, self.device, "eval")
                    if self.hinge_type == "dist":
                        batch_loss = hinge_dist(batch_emission_matrix, events, 5, self.device, "eval")

                    #print("\t" * 4, "Evaluation, batch {} loss {}".format(no_batch, batch_loss))
                    total_loss += batch_loss.item()

        print("TOTAL EVAL LOSS BEFORE DIVIDING WITH", total_loss, pass_counter, total_loss / pass_counter)
        return total_loss / pass_counter



    def training_epochs(self, corpus):


        for epoch in range(1, self.epochs+1):
            print("\t Epoch ", epoch)
            epoch_total_loss = self.train_step(corpus)
            #self.epoch_losses.append(epoch_total_loss)
            train_loss, val_loss = 0, 0

            # evaluate
            train_loss = self.evaluate(corpus.iter_batches(
                corpus.train_sents, corpus.train_events, batch_size=self.batch_size,
                seq_len=self.bptt, bptt=self.bptt, device=self.device))

            val_loss = self.evaluate(corpus.iter_batches(
                corpus.valid_sents, corpus.valid_events, batch_size=self.batch_size,
                seq_len=self.bptt, bptt=self.bptt, device=self.device))

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)

            self.lr = self.lr * self.anneal
            print("Current LR", self.lr)


    def plot_curves(self):

        x = np.arange(1, len(self.train_losses)+1)
        plt.plot(x, self.train_losses, label="Train")
        plt.plot(x, self.val_losses, label="Validation")

        plt.ylabel("Hinge loss")
        plt.xlabel("Epoch")
        plt.legend()

        plt.title("Joint event LM training (library)")
        plt.savefig("library_curves.png")

