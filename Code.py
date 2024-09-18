# !pip install transformers datasets

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import math
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tabulate import tabulate
from datasets import load_dataset
import random
from tqdm.notebook import tqdm
from transformers import BertTokenizer
import pandas as pd
import os
from functools import partial

dataset = load_dataset("scikit-learn/imdb", split="train")
print(dataset)

print(dataset.shape) # we have 50000 texts/sentences and the associated score
print(dataset[1]["review"], "\n", dataset[1]["sentiment"]) # we can have either positive and negative sentiment

"""# Pre-processing / Tokenization

"""

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)

VOCSIZE = tokenizer.vocab_size # size of the vocabulary

def preprocessing_fn(x, tokenizer):
    x["review_ids"] = tokenizer(
        x["review"],
        add_special_tokens = False,
        truncation = True,
        max_length = 256,
        padding = False,
        return_attention_mask = False,
    )["input_ids"]
    x["label"] = 0 if x["sentiment"] == "negative" else 1
    return x

print(preprocessing_fn(dataset[10], tokenizer)) # we are just interested in the columns ("input_ids", "label")
# Basically, each word/token is associated to a certain id inside the vocabulary

"""# DATA PREPROCESSING

"""

n_samples = 80  # the number of training example

# We first shuffle the data !
dataset_mod = dataset.shuffle(seed = 123) # it is very importand to shuffle the dataset in those type of applications

# Select 5000 samples
random.seed(12345)
dataset_mod = dataset_mod.select(random.sample(range(1, dataset.num_rows), n_samples)) # select randomly 5000 sentences inside the dataset

# Tokenize the dataset
dataset_mod = dataset_mod.map(preprocessing_fn, fn_kwargs = {"tokenizer": tokenizer})

# Remove useless columns
dataset_mod = dataset_mod.remove_columns(["review", "sentiment"])

# Split the train and validation
dataset_mod = dataset_mod.train_test_split(test_size = 0.2, seed = 123)

document_train_set = dataset_mod["train"]
document_valid_set = dataset_mod["test"]

n_train = len(document_train_set)
n_test = len(document_valid_set)
print("\n", f"The length of the training set is {n_train}","\n", f"The length of the testing set is {n_test}")

"""One simple idea could be to add zeros (a sort of padding) each time we are working with a word that does not allow to take into a account all the $2R$ positive contexts words. For example, if we are working with the first word of a sentence with a radius equal to R, we will the real ids and then R zeros in the remaining R positions. We understand that it is not the most powerfull idea."""

def extract_words_contexts(R, ids):
  # R: radius
  # ids: list of ids representing a sentence
  w = []
  positive_list = []

  for i in range(len(ids)):
    w.append(ids[i]) # add the corresponding word

    if i == 0: # for the first word, we just sample the next R words
      circ = ids[1:(i+R+1)]
      circ = circ + [0]*(2*R - len(circ)) # padding
      positive_list.append(circ)

    if i == (len(ids) - 1): # for the last word, we just sample the previous R words
      circ1 = ids[(i-R):i]
      circ1 = circ1 + [0]*(2*R - len(circ1)) # padding
      positive_list.append(circ1)

    elif (i!= 0 & i!= (len(ids) - 1)):
      if i < R:
        circ3 = ids[0 : i]
        circ4 = ids[(i + 1): i + R + 1]
        circ5 = circ3 + circ4
        circ5 = circ5 + [0]*(2*R - len(circ5)) # padding
        positive_list.append(circ5)
      if i >= R:
        circ3 = ids[i - R:i]
        circ4 = ids[i + 1: min(i + R + 1, len(ids))]
        circ5 = circ3 + circ4
        circ5 = circ5 + [0]*(2*R - len(circ5)) # padding
        positive_list.append(circ5)

  return w, positive_list



radius = 5
example = extract_words_contexts(R = radius, ids = document_train_set[0]["review_ids"])
print(f"For the id {example[0][2]}, the corresponding positive context with a radius {radius} is {example[1][2]}")

radius1 = 4
radius2 = 6

def flatten_dataset_to_list(data, R):
    words = []
    positive_context = []
    for input_ids in tqdm(data): # for each document
      ids = input_ids["review_ids"] # extract ids
      wo, po = extract_words_contexts(R = R, ids = ids) # extract the list with id and the list with context
      words += wo
      positive_context += po
    return words, positive_context

# In the literature it has been studied that the optimal value for the radius is between 4 and 5
doc_train = flatten_dataset_to_list(document_train_set, R = radius1) # model with radius = 4
doc_test = flatten_dataset_to_list(document_valid_set, R = radius1)

doc_train2 = flatten_dataset_to_list(document_train_set, R = radius2) # model with radius equal to 6
doc_test2 = flatten_dataset_to_list(document_valid_set, R = radius2)

class IMDBDataset(Dataset): # three mandatory components, self, get_item and len
    def __init__(self, data: list):

        self.dataset = data

    def __len__(self):

        return len(self.dataset[0])

    def __getitem__(self, idx: int):

        return self.dataset[0][idx], self.dataset[1][idx]

train = IMDBDataset(doc_train) # train with radius equal to 4
test = IMDBDataset(doc_test)

train2 = IMDBDataset(doc_train2) # train with radius equal to 6
test2 = IMDBDataset(doc_test2)

print(train[1]) # print the second word
print(len(train)) # number of words in the train dataset (not vocabulary)

"""-  negative context to the batch"""

scale1 = 2
scale2 = 4

def collate_fn(batch, K, R = radius, sizes = VOCSIZE):
  # K: scale factor for negative context
  # R: radius
  # batch is a tuple made of an integer and a list of integers
  word_id = [element[0] for element in batch]
  positive_context_ids = [element[1] for element in batch]
  neg_context_ids = []
  # we would like to sample 2K*R random words from the vocabulary
  for i in range(0, len(batch)): # for each word in the batch
    inner_negative = [] # define a list
    while len(inner_negative) < 2*K*R:
      negative = random.randint(0, sizes-1)

      while (negative == word_id[i] or negative in positive_context_ids[i]): # we would like to have word which is the different
      # from word_id and any other word in the positive context
        negative = random.randint(0, sizes-1)

      inner_negative.append(negative)

    neg_context_ids.append(inner_negative)

  return{"word_id":torch.tensor(word_id),
         "positive_context_ids": torch.tensor(positive_context_ids),
         "negative_context_ids": torch.tensor(neg_context_ids)}

"""Now we will define 4 different partial collate function with different values of the hyperparameters."""

partial1 = partial(collate_fn, K = scale1, R = radius1) #scale 2, radius 4
partial2 = partial(collate_fn, K = scale1, R = radius2) #scale 2, radius 6
partial3 = partial(collate_fn, K = scale2, R = radius1) #scale 4, radius 4
partial4 = partial(collate_fn, K = scale2, R = radius2) #scale 4, radius 6

"""Our idea is to train $8$ different models in order to make the final ablation study."""

batch_size = 256 # we decide to fix this particular batch_size in order to speed up computations
#####
# scale = 2, radius = 4
train_dataloader = DataLoader(
    train, batch_size = batch_size, collate_fn = partial1
)

valid_dataloader = DataLoader(
    test, batch_size = batch_size, collate_fn = partial1
)
n_train = len(train)
n_test = len(test)

print(f"The length of the training set is {n_train}, while the length of the test set is {n_test}")

####
# scale = 2, radius = 6
train_dataloader1 = DataLoader(
    train2, batch_size = batch_size, collate_fn = partial2
)

valid_dataloader1 = DataLoader(
    test2, batch_size = batch_size, collate_fn = partial2
)


####
# scale = 4, radius = 4

train_dataloader2 = DataLoader(
    train, batch_size = batch_size, collate_fn = partial3
)

valid_dataloader2= DataLoader(
    test, batch_size = batch_size, collate_fn = partial3
)

####
# scale = 4, radius = 6
train_dataloader3 = DataLoader(
    train2, batch_size = batch_size, collate_fn = partial4
)

valid_dataloader3 = DataLoader(
    test2, batch_size = batch_size, collate_fn = partial4
)

"""# MODEL

"""

class Word2vec(nn.Module):
  '''
  Implementation of Word2Vec from scratch using pytorch:
  - vocab_size --> size of the vocabulary
  - embedding_dim --> dimension of the embedding space --> the dimension is the same for the word and the context, but we use two different embedding tables
  We compute the scores between the words and the contexts and then we apply a sigmoid function to each cell, as the implementation requires.
  '''
  def __init__(self, vocab_size, embedding_dim):
    super().__init__()
    self.embedding_dim = embedding_dim
    self.size = vocab_size
    self.embedding = nn.Embedding(num_embeddings = self.size, embedding_dim = self.embedding_dim) # word embedding table
    self.embedding_cont = nn.Embedding(num_embeddings = self.size, embedding_dim = self.embedding_dim) # contexts embedding table

  def forward(self, input):
    self.words = self.embedding(input["word_id"]).unsqueeze(-1) #batch_size x embeddings x 1
    self.positive = self.embedding_cont(input["positive_context_ids"]) #batch_size x 2*R x embedddings
    self.negative = self.embedding_cont(input["negative_context_ids"]) #batch_size x 2*k*R x embeddings

    mult1 = torch.bmm(self.positive, self.words).squeeze(-1) # compute the score between words and the positive context
    mult2 = torch.bmm(self.negative, self.words).squeeze(-1) # compute the score between words and the negative context
    # Then apply the sigmoid function.
    sig1 = torch.sigmoid(mult1) #batch size x 2*R
    sig2 = torch.sigmoid(mult2) # bact size x 2*R*K

    return (sig1, sig2)

dim = [25, 100] # embeddings

"""We decide to test two different embeddings dimensions. So, at the end we will have $8$ different models"""

class Personal_loss(nn.Module):
  '''
  Compute the loss of the Word2vec model:
  for each observation, we sum -log(p) for the words in the positive context and -log(1-p) for the negative context.
  Then we compute the sum of the negative likelihood for each observation to have an absolute loss value for the batch.
  Our goal is to minimize the negative log-likelihood.

  We choose to add epsilon in order to avoig log of zero and have an infinitive loss
  '''

  def __init__(self, positive, negative, epsilon = 0.05):
    super().__init__()
    self.pos = positive
    self.neg = negative
    self.epsilon = epsilon

  def forward(self):
    t1 = torch.log(self.pos + self.epsilon).sum(dim = 1)
    t2 = torch.log(1 - self.neg + self.epsilon).sum(dim = 1)
    t3 = -1*t1 -1*t2

    return (torch.sum(t3))

model99 = Word2vec(vocab_size =  VOCSIZE,
                 embedding_dim = 5) # we choose 5 embeddings dimension by default and the vocsize of the bert tokenizer
out = model99(batch)
print(out[1].size())

Personal_loss(out[0], out[1]).forward().item()

"""Remark:
You will see $8$ different models with different combinations for the parameters.


We define a sort of accuracy:
- we compute the score between a word and the two contexts;
- if the score for each couple is greater than a certain threshold (fixed to 0.3) we assing the label $1$, otherwise $0$;
- words in the positive contexts have label $1$ while words in the negative contexts have label $0$.
This accuracy should state how much the model is able to understand that a word is similar to words in the positive context.

We had also the idea to implement cosine similarity but it was not a proper accuracy score.
"""

# scale = 2, radius = 4 --> train_dataloader, valid_dataloader
# scale = 2, radius = 6 --> train_dataloader1, valid_dataloader1
# scale = 4, radius = 4 --> train_dataloader2, valid_dataloader2
# scale = 4, radius = 6 --> train_dataloader3, valid_dataloader3

n_epochs = 3 # we fix a low number of epochs
def validation(mod, valid, rad, sca, thresh):
  number = 0
  correct = 0
  mod.eval()
  with torch.no_grad():

    for batch in tqdm(valid):
      number += len(batch["word_id"]) * (2*rad + 2*rad*sca) #basically we have 256x4xr^2xsca elements
      # The idea is that we wil not use just the length of a batch, because for each word you have associated a vector of size
      # 2R * 2RK
      out = mod(batch)
      pos = out[0]
      neg = out[1]
      correct += (pos >= thresh).sum().item() + (neg <= thresh).sum().item()

    print(f"The validation accuracy is {correct/number}")




def training(mod, train, epochs, rad, sca, thresh, valid):
  optimizer = torch.optim.Adam(mod.parameters(), lr = 0.001)
  for e in range(epochs):
    mod.train()
    running_loss = 0
    number = 0 # number of training elements
    correct = 0

    for batch in tqdm(train): # the batch_size is fixed to 64 here
      number += len(batch["word_id"]) * (2*rad +2*rad*sca)
      optimizer.zero_grad()
      out = mod(batch)
      pos = out[0]
      neg = out[1]
      loss = Personal_loss(pos, neg)
      loss_v = loss.forward()
      loss_v.backward()
      optimizer.step()
      running_loss += loss_v.item()
      correct += (out[0] >= thresh).sum().item() + (out[1] <= thresh).sum().item()

    print(f"Loss:{running_loss/number}")
    print(f"Accuracy:{correct/number}")
  validation(mod, valid, rad, sca, thresh)

model = Word2vec(vocab_size =  VOCSIZE,
                 embedding_dim = dim[0]) # we choose 25 embeddings dimension by default and the vocsize of the bert tokenizer
training1 = training(mod = model, train = train_dataloader, epochs = n_epochs, rad = radius1, sca = scale1, thresh = 0.3, valid = valid_dataloader)
# scale = 2, radius = 4, dim = 25

model2 = Word2vec(vocab_size =  VOCSIZE,
                 embedding_dim = dim[1]) # we choose 100 embeddings dimension by default and the vocsize of the bert tokenizer
training2 = training(mod = model2, train = train_dataloader, epochs = n_epochs, rad = radius1, sca = scale1, thresh = 0.3, valid = valid_dataloader)
 # scale = 2, radius = 4, dim = 100

model3 = Word2vec(vocab_size =  VOCSIZE,
                 embedding_dim = dim[0])
training3 = training(mod = model3, train = train_dataloader1, epochs = n_epochs, rad = radius2, sca = scale1, thresh = 0.3,  valid = valid_dataloader1)
 # scale = 2, radius = 6, dim = 25

model4 = Word2vec(vocab_size =  VOCSIZE,
                 embedding_dim = dim[1])
training4 = training(mod = model4, train = train_dataloader1, epochs = n_epochs, rad = radius2, sca = scale1, thresh = 0.3, valid = valid_dataloader1)
 # scale = 2, radius = 6, dim = 100

model5 = Word2vec(vocab_size =  VOCSIZE,
                 embedding_dim = dim[0])
training5 = training(mod = model5, train = train_dataloader2, epochs = n_epochs, rad = radius1, sca = scale2, thresh = 0.3, valid = valid_dataloader2)
# scale = 4, radius = 4, dim = 25

model6 = Word2vec(vocab_size =  VOCSIZE,
                 embedding_dim = dim[1])
training5 = training(mod = model6, train = train_dataloader2, epochs = n_epochs, rad = radius1, sca = scale2, thresh = 0.3, valid = valid_dataloader2)
# scale = 4, radius = 4, dim = 100

model7 = Word2vec(vocab_size =  VOCSIZE,
                 embedding_dim = dim[0])
training7 = training(mod = model7, train = train_dataloader3, epochs = n_epochs, rad = radius2, sca = scale2, thresh = 0.3, valid = valid_dataloader3)
# scale = 4, radius = 6, dim = 25

model8 = Word2vec(vocab_size =  VOCSIZE,
                 embedding_dim = dim[1])
training8 = training(mod = model8, train = train_dataloader3, epochs = n_epochs, rad = radius2, sca = scale2, thresh = 0.3, valid = valid_dataloader3)
# scale = 4, radius = 6, dim = 100

"""Although all the models have a good performance in testing, we cannot say that those models are good enough because this is just a **vanilla accuracy**. This could also depend on the fact that we have few samples in the validation set.

We can extrat the embeddings using this line of code
"""

embeddingss = model2.embedding # embeddings extraction
embeddingss

# DIM = 25
save_model(model, d = dim[0], R = radius1, K = scale1, B = batch_size, E = n_epochs)  # scale = 2. radius = 4
save_model(model3, d = dim[0], R = radius2, K = scale1, B = batch_size, E = n_epochs)  # scale = 2, radius = 6
save_model(model5, d = dim[0], R = radius1, K = scale2, B = batch_size, E = n_epochs) # scale = 4, radius = 4
save_model(model7, d = dim[0], R = radius2, K = scale2, B = batch_size, E = n_epochs) # scale = 4, radius = 6

# DIM = 100
save_model(model2, d = dim[1], R = radius1, K = scale1, B = batch_size, E = n_epochs)  # scale = 2. radius = 4
save_model(model4, d = dim[1], R = radius2, K = scale1, B = batch_size, E = n_epochs)  # scale = 2, radius = 6
save_model(model6, d = dim[1], R = radius1, K = scale2, B = batch_size, E = n_epochs) # scale = 4, radius = 4
save_model(model8, d = dim[1], R = radius2, K = scale2, B = batch_size, E = n_epochs) # scale = 4, radius = 6