import os
from io import open
import torch
import json
import random
import itertools

class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    
    def __init__(self, path, pad_token='<pad>', bos_token='<bos>', eos_token='<eos>'):
        self.dictionary = Dictionary()
        self.pad_idx = self.dictionary.add_word(pad_token)
        self.pad_token = pad_token
        self.bos_idx = self.dictionary.add_word(bos_token)
        self.bos_token = bos_token
        self.eos_idx = self.dictionary.add_word(eos_token)
        self.eos_token = eos_token
        # dir_new = "/home/CE/skrjanec/data_seg_all_code/library/join/train.json" or "val.json"
        self.train_sents, self.train_events = self.tokenize(os.path.join(path, "train.json"))
        self.valid_sents, self.valid_events = self.tokenize(os.path.join(path, "val.json"))
        self.test_sents, self.test_events = self.tokenize(os.path.join(path, 'val.json'))
        self.id2event = self.map(os.path.join(path, "map.json"))
    
    def tokenize(self, path):
        """
        Parameters
        ----------
        path : str : path to json file

        Returns
        -------
        data : list of lists : segments, each is a list of tokens
        event_labels : list of int : indicates event labels of segments; same order as data
        """
        # with open(path, encoding="utf8", errors="ignore") as rf:
        #     data = json.load(rf)["segments"]

        with open(path, encoding="utf8", errors="ignore") as rf:
            data_dict = json.load(rf)

        data_tuples = []
        for eventID, segments in data_dict.items():
            data_tuples += [(s, eventID) for s in segments]

        data_sorted = sorted(data_tuples, key = lambda x:len(x[0]), reverse=True)
        words = set()

        data = list()
        event_labels = []
        for (segment, evid) in data_sorted:
            segment = segment.split()
            data.append(segment)
            words.update(segment)
            event_labels.append(int(evid))

        # for segment in data:
        #     segment = segment.split()
        #     temp.append(segment)
        #     words.update(segment)
        # data = sorted(temp, key=lambda x: len(x), reverse=True)
        for word in words:
            self.dictionary.add_word(word)
        return data, event_labels

    def map(self, path_to_map):
        with open(path_to_map, "r") as jfile:
            map = json.load(jfile)
        return map

    def iter_batches(self, sents, events, batch_size=32, seq_len=32, bptt=32, device=None):
        ids = list()
        for sent in sents:
            if len(sent) >= (seq_len - 2): # account for bos and eos tokens
                sent = sent[:seq_len - 2]
            sent = [self.bos_token] + sent + [self.eos_token]
            sent = sent + [self.pad_token] * (seq_len - len(sent))
            ids.append(torch.tensor([self.dictionary.word2idx[word] for word in sent]).long().unsqueeze(0))
        # seq_len x num_sents
        ids = torch.cat(ids).t()
        bptt = min(bptt, seq_len)
        # Should not shuffle because sorted?
        for i in range(ids.size(1) // batch_size):
            # seq_len x batch_size
            batch = ids[:, i*batch_size:(i+1)*batch_size]
            batch_events = events[i*batch_size:(i+1)*batch_size]
            # slice over time dimension
            for j in range(0, batch.size(0) - 1, bptt):
                seq_len = min(bptt, batch.size(0) - 1 - j)
                data = batch[j:j+seq_len, :]
                target = batch[j+1:j+1+seq_len, :].flatten()
                #print("*********** shape of target", target.shape)
                if device is not None:
                    data = data.to(device)
                    target = target.to(device)
                yield data, target, batch_events