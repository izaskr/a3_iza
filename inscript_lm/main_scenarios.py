# https://github.com/AladdinPerzon/Machine-Learning-Collection/blob/master/ML/Pytorch/more_advanced/torchtext/torchtext_tutorial1.py
# https://pytorch.org/tutorials/beginner/text_sentiment_ngrams_tutorial.html
# https://github.com/bentrevett/pytorch-sentiment-analysis/blob/master/2%20-%20Upgraded%20Sentiment%20Analysis.ipynb
import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.data import Field, LabelField, TabularDataset, BucketIterator, Iterator
from classifier import EventClassifier_biLSTM
import time
import argparse
import numpy as np
import sys
import json
from sklearn.metrics import precision_recall_fscore_support

parser = argparse.ArgumentParser(description='Event classifier')
parser.add_argument('--id', type=str, default="3",
                    help='gpu id')
parser.add_argument('--lr', type=float, default=4.0,
                    help='learning rate')
parser.add_argument('--gamma', type=float, default=0.8,
                    help='multiplier for LR update')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='dropout rate')
parser.add_argument('--epoch', type=int, default=50,
                    help='number of epochs')
parser.add_argument('--hidden', type=int, default=256,
                    help='number of hidden units')
parser.add_argument('--scenario', type=str, default="grocery",
                    help='scenario name; one of the following: train, bicycle, grocery, tree, bus, flight, haircut, bath, cake, library')
parser.add_argument('--pretrained', type=str, default="glove.6B.50d",
                    help='pretrained word embeddings: glove.6B.50d, glove.6B.100d, glove.6B.200d, glove.6B.300d, fasttext.en.300d, fasttext.simple.300d')
parser.add_argument('--weight', type=str, default="yes",
                    help='use class weights in loss computation: use max, inverse, simple or no')
parser.add_argument('--normclass', action='store_true',
                    help='normalize class weights for loss')
parser.add_argument('--wdelta', type=float, default=1.0,
                    help='multiplier for class weights; default 1.0')

args = parser.parse_args()
GPUID = args.id
SCENARIO = args.scenario
N_EPOCHS = args.epoch
MAX_LEN = 50 # max segment legth; longer segments will be shortened to 50 tokens
LR = args.lr
GAMMA = args.gamma
VECTORS = args.pretrained

scnames = {'train', 'bicycle', 'grocery', 'tree', 'bus', 'flight', 'haircut', 'bath', 'cake', 'library'}
if SCENARIO not in scnames:
    sys.exit("--scenario has to be in " + str(scnames))


supported = {'charngram.100d', 'fasttext.en.300d', 'fasttext.simple.300d', 'glove.42B.300d', 'glove.840B.300d',
                   'glove.twitter.27B.25d', 'glove.twitter.27B.50d', 'glove.twitter.27B.100d', 'glove.twitter.27B.200d', 'glove.6B.50d',
                   'glove.6B.100d', 'glove.6B.200d', 'glove.6B.300d'}

if VECTORS not in supported:
    sys.exit("--pretrained has to be be in " + str(supported))

device = torch.device("cuda:"+GPUID if torch.cuda.is_available() else "cpu")


# get dimensionality of word embeddings given arg
EMBED_DIM = int(VECTORS.split(".")[-1][:-1])

# define how to load the data
tokenize = lambda x : x.split()

segment_text = Field(sequential=True, use_vocab=True, tokenize=tokenize, lower=True, include_lengths=True, fix_length=MAX_LEN)
gold_event = LabelField() # using Field will add an extra <unk> to the set, sequential to None
fields = {"segment": ("text", segment_text), "gold_event": ("label", gold_event)}

# load data from json
train_data, val_data, test_data = TabularDataset.splits(path="/home/CE/skrjanec/data_seg_all/" + SCENARIO +"/join",
                                                        train="train_line.json",
                                                        validation="val_line.json",
                                                        test="val_line.json",
                                                        format="json", fields=fields)


# started with glove.6B.50d
# next fasttext.en.300d
segment_text.build_vocab(train_data, max_size=100000, vectors=VECTORS, unk_init=torch.Tensor.normal_)
gold_event.build_vocab(train_data)

# counts and frequency of classes in the train set
# gold_event.vocab.freqs is a Counter object
# divide every count with the largest count to get the weight for class_i
# other options for weight calculation https://discuss.pytorch.org/t/what-is-the-weight-values-mean-in-torch-nn-crossentropyloss/11455/10
print("class count in train data", gold_event.vocab.freqs)
count_max = max(gold_event.vocab.freqs.values())

# the weights should be a torch tensor
weights = []
weights2 = []
weights3 = []
for lbl, count in gold_event.vocab.freqs.items():
    weights.append(count_max/count)
    weights2.append(1/count)
    if lbl == 0:
        weights3.append(0.1)
    else:
        weights3.append(1.0)



def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return list(np.exp(x) / np.sum(np.exp(x), axis=0)) # return a list, not a numpy array


if args.normclass:
    weights = softmax(weights)
    weights2 = softmax(weights2)
    weights = torch.tensor(weights, requires_grad=False, dtype=torch.float32).to(device)
    weights2 = torch.tensor(weights2, requires_grad=False, dtype=torch.float32).to(device)

else:
    weights = torch.tensor(weights, requires_grad=False).to(device) * args.wdelta # change magnitude of class weights
    weights2 = torch.tensor(weights2, requires_grad=False).to(device)
    weights3 = torch.tensor(weights3, requires_grad=False).to(device)

print("weights for classes; max count counts / class count", weights, gold_event.vocab.freqs.keys())
print("weights for classes; 1 / class count", weights2)
print("weights; simply penalizing event 0", weights3)



#ValueError: Got string input vector glove.6B.50d.txt, but allowed pretrained vectors are
#['charngram.100d', 'fasttext.en.300d', 'fasttext.simple.300d', 'glove.42B.300d', 'glove.840B.300d',
# 'glove.twitter.27B.25d', 'glove.twitter.27B.50d', 'glove.twitter.27B.100d', 'glove.twitter.27B.200d', 'glove.6B.50d',
# 'glove.6B.100d', 'glove.6B.200d', 'glove.6B.300d']

train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
    (train_data, val_data, test_data), batch_size=32, device=device, shuffle=True, sort_within_batch=False, sort_key=lambda a : -1*len(a.text))

# Hyperparameters
BATCH_SIZE = 32
#EMBED_DIM = 300 # NOTE change accordingly
VOCAB_SIZE = len(segment_text.vocab)
NUM_CLASSES = len(gold_event.vocab)
HIDDEN_DIM = args.hidden
N_LAYERS = 2
BIDIRECTIONAL = True
DROPOUT = args.dropout
PAD_IDX = segment_text.vocab.stoi[segment_text.pad_token]
print("vocab size", VOCAB_SIZE, "number of events", NUM_CLASSES)

#model = EventClassifier(VOCAB_SIZE, EMBED_DIM, NUM_CLASSES).to(device)
model = EventClassifier_biLSTM(VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM, NUM_CLASSES, N_LAYERS, BIDIRECTIONAL, DROPOUT, PAD_IDX)

pretrained_embeddings = segment_text.vocab.vectors
model.embedding.weight.data.copy_(pretrained_embeddings)

# zero out the unknown and padding token vectors in the model
UNK_IDX = segment_text.vocab.stoi[segment_text.unk_token]
model.embedding.weight.data[UNK_IDX] = torch.zeros(EMBED_DIM)
model.embedding.weight.data[PAD_IDX] = torch.zeros(EMBED_DIM)

# define optimizer, scheduler and loss
optimizer = optim.Adam(model.parameters(), lr=LR)
scheduler = optim.lr_scheduler.StepLR(optimizer, 1, gamma=GAMMA)

criterion = nn.CrossEntropyLoss()
# https://discuss.pytorch.org/t/passing-the-weights-to-crossentropyloss-correctly/14731/8
if args.weight in {"max", "Max"}:
    criterion = nn.CrossEntropyLoss(weight=weights, reduction="mean")
    print("... using weights max")
if args.weight in {"inverse", "Inverse"}:
    criterion = nn.CrossEntropyLoss(weight=weights2, reduction="mean")
    print("... using weights inverse")
if args.weight in {"simple", "Simple"}:
    criterion = nn.CrossEntropyLoss(weight=weights3, reduction="mean")

model = model.to(device)
criterion = criterion.to(device)

def categorical_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """
    max_preds = preds.argmax(dim = 1, keepdim = True) # get the index of the max probability
    correct = max_preds.squeeze(1).eq(y)
    return correct.sum() / torch.FloatTensor([y.shape[0]])

def PRF1(preds, y):
    """
    calculate precision, recall and f1 score. average as macro
    """
    # first put the tensors onto cpu and then numpy
    max_preds = preds.argmax(dim=1, keepdim=True)  # get the index of the max probability
    max_preds = max_preds.to("cpu")
    max_preds = max_preds.numpy()
    y = y.to("cpu")
    y = y.numpy()

    p_r_f_s = precision_recall_fscore_support(max_preds, y, average="macro")
    return p_r_f_s




def train(model, iterator, optimizer, scheduler,criterion):
    epoch_loss = 0
    epoch_acc = 0

    model.train()

    for batch in iterator:
        optimizer.zero_grad()
        #import pdb; pdb.set_trace
        text, text_lengths = batch.text
        #print("........... TEXT", text) #
        predictions = model(text, text_lengths).squeeze(1)

        loss = criterion(predictions, batch.label)
        acc = categorical_accuracy(predictions, batch.label)

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()
    # adjust the LR
    scheduler.step()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)



def evaluate(model, iterator, criterion):
    epoch_loss = 0
    epoch_acc = 0
    model.eval()

    with torch.no_grad():
        for batch in iterator:
            text, text_lengths = batch.text
            predictions = model(text, text_lengths).squeeze(1)

            loss = criterion(predictions, batch.label)
            acc = categorical_accuracy(predictions, batch.label)
            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs



best_valid_loss = float('inf')
best_valid_epoch = -1

for epoch in range(N_EPOCHS):
    start_time = time.time()

    train_loss, train_acc = train(model, train_iterator, optimizer, scheduler, criterion)
    valid_loss, valid_acc = evaluate(model, valid_iterator, criterion)

    end_time = time.time()
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), SCENARIO + '_event_classifier_bestvalid.pt')
        best_valid_epoch = epoch

    print(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc * 100:.2f}%')

#import pdb; pdb.set_trace()




def test_(model, iterator, criterion):
    epoch_loss = 0
    epoch_acc = 0
    p, r, f1, sup = [], [], [], []
    model.eval()

    with torch.no_grad():
        for batch in iterator:
            text, text_lengths = batch.text
            predictions = model(text, text_lengths).squeeze(1)

            loss = criterion(predictions, batch.label)
            acc = categorical_accuracy(predictions, batch.label)
            prfs = PRF1(predictions, batch.label)
            p.append(prfs[0])
            r.append(prfs[1])
            f1.append(prfs[2])
            epoch_loss += loss.item()
            epoch_acc += acc.item()

    avg_p = sum(p) / len(iterator)
    avg_r = sum(r) / len(iterator)
    avg_f1 = sum(f1) / len(iterator)
    return epoch_loss / len(iterator), epoch_acc / len(iterator), [avg_p, avg_r, avg_f1]

# load the model with lowest loss on val
model.load_state_dict(torch.load(SCENARIO + '_event_classifier_bestvalid.pt'))
test_loss1, test_acc1, scores = test_(model, test_iterator, criterion)
print(f'------------- Test Loss1: {test_loss1:.3f} | Test Acc1: {test_acc1*100:.2f}%')
print("-------------- Precision, recall, f, support", scores)


# write the results into a file together with the hyperparams
hyp = {"epochs":N_EPOCHS, "best_valid_epoch":best_valid_epoch, "LR":LR, "gamma":GAMMA, "hidden_dim":HIDDEN_DIM,
       "pretrained_emb":VECTORS, "dropout":DROPOUT, "n_hid_layers":N_LAYERS, "loss_class_weights":args.weight,
       "wdelta":args.wdelta, "number_of_events":NUM_CLASSES}

acc_rounded = str(round(test_acc1*100, 4))
p_rounded = str(round(scores[0], 4))
r_rounded = str(round(scores[1], 4))
f1_rounded = str(round(scores[2], 4))

def write_results(scname, hyperparams, acc, p, r, f1):
    """
    scname : str : scenario name
    hyperparams : dict : hyperparameters
    acc : str : rounded classification accuracy
    p : str : rounded macro precision
    r : str : rounded macro recall
    f1 : str : rounded macro F1

    Appends into file with results
    Returns : None
    """
    hyperparams = json.dumps(hyperparams)
    with open(scname + "_results.txt", "a") as f:
        f.write("*"*10 + "\n")
        f.write(hyperparams + "\n")
        f.write("accuracy " + acc + "% \n")
        f.write("macro-precision " + p + "\n")
        f.write("macro-recall " + r + "\n")
        f.write("macro-f1 " + f1 + "\n "*2)

    print("... wrote into file")

write_results(SCENARIO, hyp, acc_rounded, p_rounded, r_rounded, f1_rounded)


"""

#todo get p(e|x) for the train and val data: new loading is probably needed because we don't want shuffling
# redefine fields
tokenize = lambda x : x.split()
segment_text = Field(sequential=True, use_vocab=True, tokenize=tokenize, lower=True, include_lengths=True, fix_length=MAX_LEN)
gold_event = LabelField() # using Field will add an extra <unk> to the set, sequential to None
fields = {"segment": ("text", segment_text), "gold_event": ("label", gold_event)}

# TODO: current test set is the val set
fresh_data = TabularDataset(path="/home/CE/skrjanec/data_seg_all/" + SCENARIO + "/join/val_line.json",
                                                        format="json", fields=fields)

segment_text.build_vocab(fresh_data, max_size=100000, vectors="glove.6B.50d", unk_init=torch.Tensor.normal_)
gold_event.build_vocab(fresh_data)


model.load_state_dict(torch.load(SCENARIO + '_event_classifier_bestvalid.pt'))
# https://www.programmersought.com/article/7283735573/
fresh_iterator = Iterator(fresh_data, batch_size=len(fresh_data), device="cuda", shuffle=False, sort=False, sort_within_batch=False, repeat=False)

def test_predict(model, iterator, criterion):
    epoch_loss, epoch_acc = 0, 0
    model.eval()
    with torch.no_grad():
        for batch in iterator:
            text, text_lengths = batch.text
            predictions = model(text, text_lengths).squeeze(1)
            print("\t \t Predictions for the test", predictions.shape, predictions)
            loss = criterion(predictions, batch.label)
            m = torch.nn.functional.log_softmax(predictions, dim=1)
            torch.save(m, SCENARIO + '_logsoftmax_p_e_given_x.pt')
            #import pdb; pdb.set_trace()
            acc = categorical_accuracy(predictions, batch.label)
            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator), m

test_loss, test_acc, conditional_logprobs = test_predict(model, fresh_iterator, criterion)
print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%')

"""
