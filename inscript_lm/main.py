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

parser = argparse.ArgumentParser(description='Event classifier')
parser.add_argument('--lr', type=float, default=4.0,
                    help='learning rate')
parser.add_argument('--gamma', type=float, default=0.8,
                    help='multiplier for LR update')
parser.add_argument('--epoch', type=int, default=50,
                    help='number of epochs')

args = parser.parse_args()
N_EPOCHS = args.epoch
MAX_LEN = 40
LR = args.lr
GAMMA = args.gamma

# define how to load the data

tokenize = lambda x : x.split()

segment_text = Field(sequential=True, use_vocab=True, tokenize=tokenize, lower=True, include_lengths=True, fix_length=MAX_LEN)
gold_event = LabelField() # using Field will add an extra <unk> to the set, sequential to None
fields = {"segment": ("text", segment_text), "gold_event": ("label", gold_event)}

# start with library
train_data, val_data, test_data = TabularDataset.splits(path="/home/CE/skrjanec/data_seg_all_code/library/join",
                                                        train="train_line.json",
                                                        validation="val_line.json",
                                                        test="val_line.json",
                                                        format="json", fields=fields)

segment_text.build_vocab(train_data, max_size=100000, vectors="glove.6B.50d", unk_init=torch.Tensor.normal_)
gold_event.build_vocab(train_data)

#ValueError: Got string input vector glove.6B.50d.txt, but allowed pretrained vectors are
#['charngram.100d', 'fasttext.en.300d', 'fasttext.simple.300d', 'glove.42B.300d', 'glove.840B.300d',
# 'glove.twitter.27B.25d', 'glove.twitter.27B.50d', 'glove.twitter.27B.100d', 'glove.twitter.27B.200d', 'glove.6B.50d',
# 'glove.6B.100d', 'glove.6B.200d', 'glove.6B.300d']

train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
    (train_data, val_data, test_data), batch_size=32, device="cuda", shuffle=True, sort_within_batch=False, sort_key=lambda a : -1*len(a.text))

# Hyperparameters
BATCH_SIZE = 32
EMBED_DIM = 50
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VOCAB_SIZE = len(segment_text.vocab)
NUM_CLASSES = len(gold_event.vocab)
HIDDEN_DIM = 256
N_LAYERS = 2
BIDIRECTIONAL = True
DROPOUT = 0.5
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

model = model.to(device)
criterion = criterion.to(device)

def categorical_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """
    max_preds = preds.argmax(dim = 1, keepdim = True) # get the index of the max probability
    correct = max_preds.squeeze(1).eq(y)
    return correct.sum() / torch.FloatTensor([y.shape[0]])




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

for epoch in range(N_EPOCHS):
    start_time = time.time()

    train_loss, train_acc = train(model, train_iterator, optimizer, scheduler, criterion)
    valid_loss, valid_acc = evaluate(model, valid_iterator, criterion)

    end_time = time.time()
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'event_classifier_bestvalid.pt')

    print(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc * 100:.2f}%')

#import pdb; pdb.set_trace()

#todo get p(e|x) for the train and val data: new loading is probably needed because we don't want shuffling
# redefine fields
tokenize = lambda x : x.split()
segment_text = Field(sequential=True, use_vocab=True, tokenize=tokenize, lower=True, include_lengths=True, fix_length=MAX_LEN)
gold_event = LabelField() # using Field will add an extra <unk> to the set, sequential to None
fields = {"segment": ("text", segment_text), "gold_event": ("label", gold_event)}

fresh_data = TabularDataset(path="/home/CE/skrjanec/data_seg_all_code/library/join/train_val_line.json",
                                                        format="json", fields=fields)

segment_text.build_vocab(fresh_data, max_size=100000, vectors="glove.6B.50d", unk_init=torch.Tensor.normal_)
gold_event.build_vocab(fresh_data)


model.load_state_dict(torch.load('event_classifier_bestvalid.pt'))
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
            torch.save(m, 'logsoftmax_p_e_given_x.pt')
            #import pdb; pdb.set_trace()
            acc = categorical_accuracy(predictions, batch.label)
            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator), m

test_loss, test_acc, conditional_logprobs = test_predict(model, fresh_iterator, criterion)
print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%')

