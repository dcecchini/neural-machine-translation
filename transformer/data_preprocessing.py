# -*- coding: utf-8 -*-
"""
Classes and functions implemented for the NMT Graph translation model
    Preprocessing of datasets

@author: David Cecchini
@author2: Steve Beattie
"""

import os
import re
import random
import numpy as np
from dgl.data.utils import download, extract_archive
from sacremoses import MosesTokenizer

from transformer_header import TranslationDataset


_urls = {
    'wmt': 'https://s3.us-east-2.amazonaws.com/dgl.ai/dataset/wmt14bpe_de_en.zip',
    'scripts': 'https://s3.us-east-2.amazonaws.com/dgl.ai/dataset/transformer_scripts.zip',
}


def preprocess_anki(filename, output_folder, input_lang='en', target_lang='pt', tolower=False, remove_punct=False):
    """Separates the anki file into files for input and target languages

    Accepts an raw .txt file with phrases separated by `\t`. Splits the languages
    and separate into train, validation and test sets. Save the files on the `output_folder`.

    Args:
        filename (str): The complete path and filnename containing the data.
        output_folder(str): The destination folder where to save the files.
        input_lang(str): Language on the first column of the anki file. Defaults to 'en' (English).
        target_lang(str): Language on the second column of the anki file. Defaults to 'pt' (Portuguese).
        tolower(bool): If the texts be converted to lower case or not.
        remove_punct(bool): If punctuation should be removed from the text or not.

    Returns:
        Nothing

    """

    # Get contents of the file
    with(open(filename, mode='rt', encoding='utf-8')) as file:
        text = file.read()

    # Transform to lower case
    if tolower:
        text = text.lower()

    # Separete lines and languages
    lines = text.strip().split('\n')
    pairs = [line.split('\t') for line in lines]

    # Remove punctuation
    if remove_punct:
        rem_punct = re.compile('\W+')
        cleaned = list()
        for pair in pairs:
            clean_pair = list()
            for line in pair:
                line = line.split(' ')
                line = [rem_punct.sub('', w) for w in line]
                # remove tokens with numbers in them
                line = [word for word in line if word.isalpha()]
                clean_pair.append(' '.join(line))
            cleaned.append(clean_pair)
        pairs = cleaned
        del cleaned

    # Separate into train, validation and test sets
    # Shuffle the data to make it a random sample
    # random.seed(2019)
    random.shuffle(pairs)

    # Create a file with 29000 obs for train, 1000 for val and 1000 for test
    train = pairs[:29000]
    val = pairs[29000:30000]
    test = pairs[30000:31000]

    for file_part, dataset in zip(['train', 'val', 'test'], [train, val, test]):
        input_text = dataset[:, 0].tolist() # [line[0] for line in dataset]
        target_text = dataset[:, 1].tolist() # [line[1] for line in dataset]
        for lang, text in zip([input_lang, target_lang], [input_text, target_text]):
            out_file = os.path.join(output_folder, "{0}.{1}".format(file_part, lang))
            with(open(out_file, "w", encoding="utf-8")) as f:
                f.write('\n'.join(text))

    return


# TODO: Adapt to our needs
def prepare_dataset(dataset_name, tolower=False, remove_punct=False):
    """download and generate datasets. Runs only once
    """
    # script_dir = os.path.join('scripts')
    # if not os.path.exists(script_dir):
    #     download(_urls['scripts'], path='scripts.zip')
    #     extract_archive('scripts.zip', 'scripts')
    import glob
    directory = os.path.join('data', dataset_name)
    if not os.path.exists(directory):
        os.makedirs(directory)
    else:
        if len(glob.glob(os.path.join(directory, "*.atok"))) > 0:
            return
    #if dataset_name == 'multi30k':
    # os.system('bash scripts/prepare-multi30k.sh')
    files = [f for f in glob.glob(os.path.join(directory, "*"))]
    for f in files:
        tokenizer = MosesTokenizer(lang=f[-2:])
        with(open(f, "r", encoding="utf-8")) as input_file:
            input_text = input_file.read()
        if tolower:
            input_text = input_text.lower()
        input_text = input_text.strip().split('\n')
        if remove_punct:
            rem_punct = re.compile('\W+')
            for i in range(len(input_text)):
                line = input_text[i].split(' ')
                line = [rem_punct.sub('', w) for w in line]
                input_text[i] = ' '.join(line)

        text_atok = [tokenizer.tokenize(text=t, aggressive_dash_splits=True, return_str=True, escape=False) for t in input_text]
        with(open(f + ".atok", "w", encoding="utf-8")) as out_file:
            out_file.write('\n'.join(text_atok))

    # elif dataset_name == 'wmt14':
    #     download(_urls['wmt'], path='wmt14.zip')
    #     os.system('bash scripts/prepare-wmt14.sh')
    # elif dataset_name == 'copy' or dataset_name == 'tiny_copy':
    #     train_size = 9000
    #     valid_size = 1000
    #     test_size = 1000
    #     char_list = [chr(i) for i in range(ord('a'), ord('z') + 1)]
    #     with open(os.path.join(directory, 'train.in'), 'w') as f_in,\
    #         open(os.path.join(directory, 'train.out'), 'w') as f_out:
    #         for i, l in zip(range(train_size), np.random.normal(15, 3, train_size).astype(int)):
    #             l = max(l, 1)
    #             line = ' '.join(np.random.choice(char_list, l)) + '\n'
    #             f_in.write(line)
    #             f_out.write(line)
    #     with open(os.path.join(directory, 'valid.in'), 'w') as f_in,\
    #         open(os.path.join(directory, 'valid.out'), 'w') as f_out:
    #         for i, l in zip(range(valid_size), np.random.normal(15, 3, valid_size).astype(int)):
    #             l = max(l, 1)
    #             line = ' '.join(np.random.choice(char_list, l)) + '\n'
    #             f_in.write(line)
    #             f_out.write(line)
    #     with open(os.path.join(directory, 'test.in'), 'w') as f_in,\
    #         open(os.path.join(directory, 'test.out'), 'w') as f_out:
    #         for i, l in zip(range(test_size), np.random.normal(15, 3, test_size).astype(int)):
    #             l = max(l, 1)
    #             line = ' '.join(np.random.choice(char_list, l)) + '\n'
    #             f_in.write(line)
    #             f_out.write(line)
    #     with open(os.path.join(directory, 'vocab.txt'), 'w') as f:
    #         for c in char_list:
    #             f.write(c + '\n')
    # elif dataset_name == 'sort' or dataset_name == 'tiny_sort':
    #     train_size = 9000
    #     valid_size = 1000
    #     test_size = 1000
    #     char_list = [chr(i) for i in range(ord('a'), ord('z') + 1)]
    #     with open(os.path.join(directory, 'train.in'), 'w') as f_in,\
    #         open(os.path.join(directory, 'train.out'), 'w') as f_out:
    #         for i, l in zip(range(train_size), np.random.normal(15, 3, train_size).astype(int)):
    #             l = max(l, 1)
    #             seq = np.random.choice(char_list, l)
    #             f_in.write(' '.join(seq) + '\n')
    #             f_out.write(' '.join(np.sort(seq)) + '\n')
    #     with open(os.path.join(directory, 'valid.in'), 'w') as f_in,\
    #         open(os.path.join(directory, 'valid.out'), 'w') as f_out:
    #         for i, l in zip(range(valid_size), np.random.normal(15, 3, valid_size).astype(int)):
    #             l = max(l, 1)
    #             seq = np.random.choice(char_list, l)
    #             f_in.write(' '.join(seq) + '\n')
    #             f_out.write(' '.join(np.sort(seq)) + '\n')
    #     with open(os.path.join(directory, 'test.in'), 'w') as f_in,\
    #         open(os.path.join(directory, 'test.out'), 'w') as f_out:
    #         for i, l in zip(range(test_size), np.random.normal(15, 3, test_size).astype(int)):
    #             l = max(l, 1)
    #             seq = np.random.choice(char_list, l)
    #             f_in.write(' '.join(seq) + '\n')
    #             f_out.write(' '.join(np.sort(seq)) + '\n')
    #     with open(os.path.join(directory, 'vocab.txt'), 'w') as f:
    #         for c in char_list:
    #             f.write(c + '\n')


def get_dataset(dataset):
    "we wrapped a set of datasets as example"
    prepare_dataset(dataset)
    if dataset == 'babi':
        raise NotImplementedError
    elif dataset == 'copy' or dataset == 'sort':
        return TranslationDataset(
            'data/{}'.format(dataset),
            ('in', 'out'),
            train='train',
            valid='valid',
            test='test',
        )
    elif dataset == 'multi30k':
        return TranslationDataset(
            'data/multi30k',
            ('en.atok', 'de.atok'),
            train='train',
            valid='val',
            test='test2016',
            replace_oov='<unk>'
        )
    elif dataset == 'anki':
        return TranslationDataset(
            'data/anki',
            ('en.atok', 'pt.atok'),
            train='train',
            valid='val',
            test='test',
            replace_oov='<unk>'
        )
    elif dataset == 'wmt14':
        return TranslationDataset(
            'data/wmt14',
            ('en', 'de'),
            train='train.tok.clean.bpe.32000',
            valid='newstest2013.tok.bpe.32000',
            test='newstest2014.tok.bpe.32000.ende',
            vocab='vocab.bpe.32000')
    else:
        raise KeyError()



