"""
Assign a loglikelihood score to a sentence and then add the log posterior probability to get the score
Then subtract the log prior prob of an event
"""

from lm_scorer.models.auto import AutoLMScorer as LMScorer
import json
import torch
from visual_heatmap import ConfusionMatrixHeatMap

class GetLoglikelihood():
    def __init__(self, scenario="library"):
        self.scenario = scenario
        self.data_path = "/home/CE/skrjanec/data_seg_all_code/" + self.scenario + "/join/train_val_line.json"

        # initialize the LM scorer
        self.init_LM_scorer()
        self.lm_ll = []

        # read the data json
        # '{"gold_event": "5", "segment": "he then scanned my card ,"}'
        self.gold_labels = []
        with open(self.data_path, "r") as f:
            print("... reading in the segments")
            for line in f:
                if line:
                    segment = json.loads(line)["segment"] # str
                    self.lm_ll.append(self.scorer.sentence_score(segment, log=True)) # log probability of sentence; float
                    self.gold_labels.append(int(json.loads(line)["gold_event"])) # str

        self.gold_labels = torch.tensor(self.gold_labels)

        self.lm_ll = torch.tensor(self.lm_ll)
        print("number of segments", self.lm_ll.shape)
        self.lm_ll = self.lm_ll.to("cpu")

        # get prior probabilities of events from a file
        self.read_priors()
        print("priors", self.priors)
        self.priors = self.priors.to("cpu")

        self.read_posteriors()
        print("posteriors", self.posteriors)
        self.posteriors = self.posteriors.to("cpu")

        # sum and subtract
        self.data_likelihood = self.posteriors.detach().clone()
        self.data_likelihood = self.data_likelihood.to("cpu")
        import pdb; pdb.set_trace()

        for i in range(self.data_likelihood.shape[1]):
            self.data_likelihood[:, i] += self.lm_ll

        for j in range(self.data_likelihood.shape[0]):
            self.data_likelihood[j, :] -= self.priors


        #self.visualize()
        #import pdb; pdb.set_trace()

        # check which event makes a segment more likely
        # take argmax for each row = each segment
        max_LL = torch.argmax(self.data_likelihood, dim=1)
        accuracy = 100 * torch.sum(max_LL == self.gold_labels).item() / self.data_likelihood.shape[0]
        print("ACCURACY", accuracy)

    def read_posteriors(self):
        tpath = "/home/CE/skrjanec/data_seg_all_code/word_language_model_iza/classifier/inscript_lm/logsoftmax_p_e_given_x.pt" # path to tensor
        self.posteriors = torch.load(tpath)
        print("Posterior logp tensor", self.posteriors)

    def init_LM_scorer(self):
        device = "cuda:1"
        batch_size = 1
        self.scorer = LMScorer.from_pretrained("gpt2", device=device, batch_size=batch_size)

        # scorer.sentence_score([s1, s2...], log=True)

    def read_priors(self):
        fpath = "/home/CE/skrjanec/data_seg_all_code/" + self.scenario + "/event_prior.json"
        with open(fpath, "r") as jf:
            self.priors = json.load(jf) # direct probabilities, not log

        # make probabilities into torch tensors and logs
        # first arrange them in right order (0, 1...) and as a tensor
        self.priors = torch.tensor([self.priors[str(j)] for j in range(len(self.priors))])
        self.priors = torch.log(self.priors)

    def visualize(self):
        # tensor, y_names, x_names, out_name
        y_names = [str(k) for k in range(1, self.data_likelihood.shape[0]+1)] # segment inidices
        x_names = [str(e) for e in range(self.priors.shape[0])]
        out_name = "dataLL_" + self.scenario
        import pdb; pdb.set_trace()

        ConfusionMatrixHeatMap(self.data_likelihood, y_names, x_names, out_name)


getll = GetLoglikelihood(scenario="library")
