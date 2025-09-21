import pandas as pd
import numpy as np
import spacy
from torchtext.data import Field, Dataset
import torchtext.data as data
import torchtext
import sys
import re

sys.path.append('..')
sys.path.append('../clai/utils')

import config
spacy_en = spacy.load('en_core_web_sm')

def tokenize_nl(text):
  return [tok.text for tok in spacy_en.tokenizer(text)][::-1]

def tokenize_bash(text):
    """
    A more robust bash command tokenizer that handles:
    - Quoted strings (both single and double quotes)
    - Command options/flags
    - File paths
    - Special characters
    - Pipes and redirections
    """
    # First, let's preserve quoted strings
    quoted_strings = []
    def preserve_quoted(match):
        quoted_strings.append(match.group(0))
        return f' QUOTEDSTR{len(quoted_strings)-1} '

    # Handle quoted strings (both single and double quotes)
    text = re.sub(r'"[^"]*"|\'[^\']*\'', preserve_quoted, text)

    # Add spaces around special characters
    special_chars = '|><;()[]{}&'
    for char in special_chars:
        text = text.replace(char, f' {char} ')

    # Handle flags/options
    text = re.sub(r'(?<!\S)(-{1,2}[a-zA-Z][a-zA-Z0-9-]*)', r' \1 ', text)

    # Split by whitespace
    tokens = text.split()

    # Restore quoted strings
    for i, token in enumerate(tokens):
        if token.startswith('QUOTEDSTR'):
            try:
                idx = int(token[9:])
                tokens[i] = quoted_strings[idx]
            except (ValueError, IndexError):
                pass

    return tokens


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