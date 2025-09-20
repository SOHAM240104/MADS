import pandas as pd
import numpy as np
import spacy
from torchtext.data import Field, Dataset
import torchtext.data as data
import torchtext
import sys

sys.path.append('..')
sys.path.append('../clai/utils')

from bashlint import data_tools
import config
spacy_en = spacy.load('en_core_web_sm')

def tokenize_nl(text):
  return [tok.text for tok in spacy_en.tokenizer(text)][::-1]

def tokenize_bash(text):
  try:
    return data_tools.bash_tokenizer(text)
  except:
    # Fallback for malformed commands - just split by spaces
    return text.split()


def create_dataset():    
    SRC = Field(
        tokenize = tokenize_nl,
        init_token='<sos>',
        eos_token='<eos>',
        lower=True
    )

    TRG = Field(
        tokenize = tokenize_bash,
        init_token='<sos>',
        eos_token='<eos>',
        lower=True
    )

    train = data.TabularDataset(
    path=config.TRAIN_PATH, format=config.X_FORMAT,
    fields=[('nl', SRC),
            ('code', TRG)])


    test = data.TabularDataset(
    path=config.TEST_PATH, format=config.X_FORMAT,
    fields=[('nl', SRC),
            ('code', TRG)])


    
    valid = data.TabularDataset(
    path=config.VALID_PATH, format=config.X_FORMAT,
    fields=[('nl', SRC),
            ('code', TRG)])


    SRC.build_vocab(train)

    TRG.build_vocab(train)

    return train, valid, test, SRC, TRG