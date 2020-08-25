Language model training with hinge loss

The hinge loss is margin-based, not likelihood


The hinge loss is used for classification
Penalizes the model for letting competitors get  too close to the correct answer y
    – Can penalize them in proportion to their error.



Some references:
- http://www.cs.cmu.edu/~nasmith/psnlp/lecture4.pdf

- https://arxiv.org/pdf/1510.00726.pdf (page 18)

- MultiMarginLoss: https://pytorch.org/docs/stable/nn.html


To download the GloVe embeddings, do the following:
wget nlp.stanford.edu/data/glove.6B.zip 
unzip glove.6B.zip

I use only the 50-dimensional ones, so glove.6B.50d.txt.
