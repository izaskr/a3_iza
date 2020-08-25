# coding: utf-8
import argparse
import time
import math
import os
import torch
import torch.nn as nn
import torch.onnx
import torch.optim as optim
from glove import load_glove_embeddings, get_glove_embeddings
import json
import data
import model_forwardweights
import hinge_loss
from language_modeling import EventLanguageModeling
import itertools

parser = argparse.ArgumentParser(description='Event RNN/LSTM/GRU/Transformer Language Model')
parser.add_argument('--data', type=str, default='data/',
                    help='location of the data corpus')
parser.add_argument('--eventid', type=str, default='1',
                    help='event ID for the library scenario')
parser.add_argument('--gpu', type=str, default='0', help='GPU ID')
parser.add_argument('--model', type=str, default='LSTM',
                    help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU, Transformer)')
parser.add_argument('--emsize', type=int, default=50,
                    help='size of word embeddings, default 50 for GloVe')
parser.add_argument('--nhid', type=int, default=70,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=2,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=7.0,
                    help='initial learning rate')
parser.add_argument('--anneal', type=float, default=0.25,
                    help='annealing rate for the learning rate; LR will be multiplied with by this factor')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=60,
                    help='upper epoch limit')
parser.add_argument('--hinge', type=str, default="max",
                    help='loss computation: max or dist')
parser.add_argument('--batch_size', type=int, default=32, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=35,
                    help='sequence length')
parser.add_argument('--dropout', type=float, default=0.3,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--tied', action='store_true',
                    help='tie the word embedding and softmax weights')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='report interval')
parser.add_argument('--save', type=str, default='best_val.pt',
                    help='path to save the final model')
parser.add_argument('--onnx-export', type=str, default='',
                    help='path to export the final model in onnx format')

parser.add_argument('--nhead', type=int, default=2,
                    help='the number of heads in the encoder/decoder of the transformer model')

args = parser.parse_args()

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

device = torch.device("cuda:"+args.gpu if args.cuda else "cpu")

###############################################################################
# Load data
###############################################################################

corpus = data.Corpus(args.data)

# for the given vocab, load the GloVe embeddings
#vocab = corpus.dictionary.word2idx
glove_embeddings_vocab = load_glove_embeddings(corpus.dictionary.word2idx)

save_vocab = False
if save_vocab:
    with open(str(args.data)+"_all_vocab.json", "w") as jf:
        json.dump(corpus.dictionary.word2idx, jf)

eval_batch_size = 10

ntokens = len(corpus.dictionary)
id2event = corpus.id2event
event_list = sorted(list(int(e) for e in id2event))


lm_obj=EventLanguageModeling(id2event, event_list, corpus, args.batch_size, args.bptt, args.bptt, args.model, ntokens, args.emsize, args.nhid, args.nlayers, glove_embeddings_vocab, corpus.pad_idx, args.dropout, args.tied, device, args.lr, args.epochs, args.anneal, args.hinge)


