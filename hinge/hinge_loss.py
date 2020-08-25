"""
Implementation of the hinge loss
"""
import torch
import numpy as np


def hinge_max(matrix_emissions, gold_events, delta, device, mode):
    """
    Hinge loss on log probabilities
    Parameters
    ----------
    matrix_emissions : tensor (events × segments) : containing LOG probabilities of segments, given by each event LM
    gold_events : list of int : for each segment in matrix_emissions, it has the gold event ID
    delta : int : margin/distance between gold prob and max non-gold prob; 1 in parsing, check 2 or 3
    mode :  str : train or eval mode to print or not

    Remember this is batch-wise

    Returns
    -------
    tensor of hinge loss
    """
    # loss = max(0, delta - (gold prob - max non-gold prob) )

    batch_hinge = torch.tensor(0.0, dtype=torch.float64, requires_grad=True, device=device)
    matrix_emissions = matrix_emissions.t()

    for segmentID, segment_logprobs in enumerate(matrix_emissions):
        # gold_events[segmentID] is the gold eventID
        gold_logp = segment_logprobs[gold_events[segmentID]] # a tensor


        # find 2 highest logprobs, if the first one is from gold, take the second one
        result = torch.topk(segment_logprobs, k = 2)
        #print("RESULT OF TOP K", result)
        best_nongoldID = result.indices[0].item()
        best_nongold_logp = segment_logprobs[best_nongoldID] # a tensor
        #best_nongold_logp = result.values[0].item()
        if best_nongoldID == gold_events[segmentID]:
            best_nongoldID = result.indices[1].item()
            #best_nongold_logp = result.values[1].item()
            best_nongold_logp = segment_logprobs[best_nongoldID]

        # compute loss for this segment
        # turn log probabilities into standard probabilities; torch log softmax uses the natural log as base
        #segment_hinge = max(0, delta - (np.exp(gold_logp.item()) - np.exp(best_nongold_logp)))

        # use log probabilities and set delta accordingly: positive, e.g. 5, 10 ...
        #segment_hinge = max(0, delta - (gold_logp.item() - best_nongold_logp))
        #segment_hinge = torch.max(torch.tensor(0.0), delta - (gold_logp - best_nongold_logp))


        # NOTE: the 0 tensor must also be on the same device as others, the default is cpu
        batch_hinge = batch_hinge + torch.max(torch.tensor(0.0, device=device), delta - (gold_logp - best_nongold_logp))
        # batch_hinge += segment_hinge
        if mode == "eval":
            print("\t"*4, "Segment loss on val data")
            print("Gold event ID", gold_events[segmentID] )
            print("Best nongold ID", best_nongoldID)
            print("Gold logprob", gold_logp)
            print("Best nongold logp", best_nongold_logp)
            print("All logprobs across events", segment_logprobs)
            print("Loss value for this segment", torch.max(torch.tensor(0.0, device=device), delta - (gold_logp - best_nongold_logp)))

    return batch_hinge / matrix_emissions.shape[0]


def hinge_dist(matrix_emissions, gold_events, delta, device, mode):
    """
    Hinge loss on log probabilities: include all non-gold probabilities in the loss computation
    Parameters
    ----------
    matrix_emissions : tensor (events × segments) : containing LOG probabilities of segments, given by each event LM
    gold_events : list of int : for each segment in matrix_emissions, it has the gold event ID
    delta : int : margin/distance between gold prob and max non-gold prob; 1 in parsing, check 2 or 3
    mode :  str : train or eval mode to print or not

    Remember this is batch-wise

    Returns
    -------
    tensor of hinge loss
    """
    # loss = max(0, delta - (gold prob - max non-gold prob) )

    batch_hinge = torch.tensor(0.0, dtype=torch.float64, requires_grad=True, device=device)
    matrix_emissions = matrix_emissions.t() # shape: segments × events

    for segmentID, segment_logprobs in enumerate(matrix_emissions):
        # gold_events[segmentID] is the gold eventID
        #gold_logp = segment_logprobs[gold_events[segmentID]] # a tensor

        goldID = gold_events[segmentID]
        non_gold_IDs = [i for i in range(len(segment_logprobs)) if i != goldID]

        batch_hinge = batch_hinge + torch.sum(torch.max(torch.zeros(segment_logprobs[non_gold_IDs].shape,device=device), delta - (segment_logprobs[goldID] - segment_logprobs[non_gold_IDs])))

        # NOTE: the 0 tensor must also be on the same device as others, the default is cpu
        #batch_hinge = batch_hinge + torch.max(torch.tensor(0.0, device=device), delta - (gold_logp - best_nongold_logp))

        if mode == "eval":
            print("\t", "Segment loss on val data")
            print("Gold event ID", goldID)
            print("all logprobs for this sentence", segment_logprobs)
            print(batch_hinge)
            print("\n")

    return batch_hinge / matrix_emissions.shape[0]




