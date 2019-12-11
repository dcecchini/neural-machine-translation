# -*- coding: utf-8 -*-
"""
RNN for language modeling: Neural Machine Translation

@author: David Cecchini
@author2: Stece Beattie
"""


import re
from keras.layers.recurrent import LSTM
from keras.initializers import Constant
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Embedding, Dense, RepeatVector, TimeDistributed, Bidirectional
import numpy as np
import pickle
from keras.callbacks import ModelCheckpoint
from nltk.translate.bleu_score import corpus_bleu
from sklearn.model_selection import train_test_split



# Hyper Parameters
WORDVEC_DIM = 100
EPOCHS = 100
BATCH_SIZE = 64
TEST_SIZE=0.2
INPUT_COLUMN = 0  # Use English as input: 0, use Portugues: 1
TARGET_COLUMN = 1 # Use English as input: 0, use Portugues: 1

MAX_VOCAB_SIZE = 5000 # Limit the vocabulary size for memory resons
OOV_TOKEN = r"<OOV>" # Out of vocabulary token


def load_anki(filename):
    """Loads the anki file and removes punctiation

    Open the Anki file with the examples of translation phrases from both languages.

    Args:
        filename (str): The complete path and filnename containing the data.

    Returns:
        A  numpy array with two columns, representing each language's phraes

    """

    # Get contents of the file
    with(open(filename, mode='rt', encoding='utf-8')) as file:
        text = file.read()

    # Transform to lower case
    text = text.lower()

    # Separete lines and languages
    lines = text.strip().split('\n')

    # Remove punctuation
    rem_punct = re.compile('\W+')
    pairs = [line.split('\t') for line in lines]
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
    return np.array(cleaned)


# fit a tokenizer
def create_tokenizer(lines, max_vocab_size=None, oov_token=None):
    """Instantiate a tokenizer and fit it on given data
    """
    tokenizer = Tokenizer(num_words=max_vocab_size, oov_token=oov_token)
    tokenizer.fit_on_texts(lines)
    return tokenizer


# encode and pad sequences
def encode_sequences(lines, tokenizer, length):
    """Use the tokenizer to encode the given texts

    Transform the given texts into an numpy array of index sequences padded to length `length`.

    Args:
        lines (list or numpy.array): The given texts.
        tokenizer (keras.preprocessing.text.Tokenizer): The fitted tokenizer object. Defaults to the trained Portuguese tokenizer.
        length (int): The length to pad the sequences. Defaults to the maximum length of the Portuguese sentences (8).

    Returns:
        The padded numpy.array of indices sequences.

    """
    # integer encode sequences
    X = tokenizer.texts_to_sequences(lines)
    # pad sequences with 0 values
    X = pad_sequences(X, maxlen=length, padding='post')
    return X


def encode_output(sequences, vocab_size):
    """ One-hot encode the target sequences
    """

    ylist = list()
    for sequence in sequences:
        encoded = to_categorical(sequence, num_classes=vocab_size)
        ylist.append(encoded)
    y = np.array(ylist)
    y = y.reshape(sequences.shape[0], sequences.shape[1], vocab_size)
    return y


# Function to get the translation
def predict_sequence(model, source, target_ix_to_word):
    prediction = model.predict(source, verbose=0)[0]
    integers = [np.argmax(vector) for vector in prediction]
    target = list()
    for i in integers:
        word = target_ix_to_word.get(i, None)
        if word is None:
            break
        target.append(word)
    return ' '.join(target)


# Function to get one translation
def predict_one(model, source, target_ix_to_word):
    source = source.reshape((1, source.shape[0]))
    prediction = model.predict(source, verbose=0)[0]
    integers = [np.argmax(vector) for vector in prediction]
    target = list()
    for i in integers:
        word = target_ix_to_word.get(i, None)
        if word is None:
            break
        target.append(word)
    return ' '.join(target)


# Predict many
def predict_many(model, sources, target_ix_to_word, raw_dataset):
    for i, source in enumerate(sources):
        # translate encoded source text
        translation = predict_one(model, source, target_ix_to_word)
        raw_target, raw_src = raw_dataset[i]
        print('src=[%s], target=[%s], predicted=[%s]' % (raw_src, raw_target, translation))


# Exercise funtion:
def translate_many(model, sentences):
    """ Translate a list of sentences

    Use the pre-trained model to loop over the sentences and translate one by one.

    Args:
        model (keras.models.Sequential): The pre-trained NMT model.
        sentences (list or numpy.array): The list of sentences to translate.

    Returns:
        A list containing the translated sentences.

    """
    translated = []
    for i, sentence in enumerate(sentences):
        # translate encoded sentence text
        translation = predict_one(model, sentence)
        translated.append(translation)

    return translated


# evaluate the skill of the model
def evaluate_model( model, new_data, input_tokenizer, input_length, target_tokenizer, target_length, print_examples=False):
    # Get the
    input_vocab_size = len(input_tokenizer.word_index) + 1 # Add one for <PAD> token
    target_vocab_size = len(target_tokenizer.word_index) + 1 # Add one for <PAD> token

    # Get dictionary {index:word}
    target_ix_to_word = {k:w for w,k in target_tokenizer.word_index.items()}

    # prepare validation data
    X_test = encode_sequences(new_data[:, INPUT_COLUMN], input_tokenizer, input_length)
    Y_test = encode_sequences(new_data[:, TARGET_COLUMN], target_tokenizer, target_length)
    Y_test = encode_output(Y_test, target_vocab_size)

    actual, predicted = list(), list()
    for i, source in enumerate(X_test):
        # translate encoded source text
        source = source.reshape((1, source.shape[0]))
        translation = predict_sequence(model, source, target_ix_to_word)
        raw_target, raw_src = raw_dataset[i]
        if print_examples:
            if i < 10:
                print('src=[%s], target=[%s], predicted=[%s]' % (raw_src, raw_target, translation))
        actual.append(raw_target.split())
        predicted.append(translation.split())

    bleu_1 = corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0))
    bleu_2 = corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0))
    bleu_3 = corpus_bleu(actual, predicted, weights=(0.3, 0.3, 0.3, 0))
    bleu_4 = corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25))

    # calculate BLEU score
    print('BLEU-1: %f' % bleu_1)
    print('BLEU-2: %f' % bleu_2)
    print('BLEU-3: %f' % bleu_3)
    print('BLEU-4: %f' % bleu_4)

    return (bleu_1, bleu_2, bleu_3, bleu_4)


def prep_anki(filename, sample=None):
    """Open the sample texts and return as numpy array.
    """

    import os

    anki_file = filename
    anki = load_anki(anki_file)

    if sample is not None:
        anki = anki[np.random.choice(anki.shape[0], sample, replace=False), :]
        filename = filename.replace(".txt", "") + "_sample_" + str(int(sample/1000)) + "k.txt"

    np.save(filename.replace(".txt", "") + "_clean.npy", anki)

    return anki


def prep_tokenizers(anki, max_vocab_size=None, oov_token=None, name_modifier="tokenizer"):
    """Prepare the tokenizer objects.
    """

    # prepare english tokenizer
    input_tokenizer = create_tokenizer(anki[:, INPUT_COLUMN])

    # prepare Portuguese tokenizer
    target_tokenizer = create_tokenizer(anki[:, TARGET_COLUMN])

    with(open('data/input_{}.pickle'.format(name_modifier), 'wb')) as f:
        pickle.dump(input_tokenizer, f)

    with(open('data/target_{}.pickle'.format(name_modifier), 'wb')) as f:
        pickle.dump(target_tokenizer, f)

    return (input_tokenizer, target_tokenizer)


def load_tokenizers(input_tok_name, target_tok_name):
    with(open(input_tok_name, 'rb')) as f:
        input_tokenizer = pickle.load(f)

    with(open(target_tok_name, 'rb')) as f:
        target_tokenizer = pickle.load(f)

    return (input_tokenizer, target_tokenizer)


def build_model(input_vocab_size, input_length, target_vocab_size, target_length):
    """Creates the keras model used for NMT to translate EN to PT
    """

    # Define the model
    model = Sequential()

    # Encoder
    model.add(Embedding(input_vocab_size, WORDVEC_DIM, input_length=input_length, mask_zero=True))
    model.add(Bidirectional(LSTM(128)))
    model.add(RepeatVector(target_length))

    # Decoder
    model.add(LSTM(256, return_sequences=True))
    model.add(TimeDistributed(Dense(target_vocab_size, activation='softmax')))

    # Compile: The loss is as a multiclass classification
    model.compile(optimizer='adam', loss='categorical_crossentropy')

    return model


def train_model(train, input_tokenizer, input_length, target_tokenizer, target_length):
    """Train the RNN model
    """

    # Get the size of the vocabularies
    input_vocab_size = len(input_tokenizer.word_index) + 1   # Add one for <PAD> token
    target_vocab_size = len(target_tokenizer.word_index) + 1 # Add one for <PAD> token

    # prepare training data
    X_train = encode_sequences(train[:, INPUT_COLUMN], input_tokenizer, input_length)
    Y_train = encode_sequences(train[:, TARGET_COLUMN], target_tokenizer, target_length)
    Y_train = encode_output(Y_train, target_vocab_size)

    # Get dictionary {index:word}
    target_ix_to_word = {k:w for w,k in target_tokenizer.word_index.items()}

    model = build_model(input_vocab_size, input_length, target_vocab_size, target_length)

    # Callback to save the model when it gets better
    # checkpoint = ModelCheckpoint("data/rnn_model_checkpoints.h5", monitor='val_loss', verbose=1, save_best_only=True, mode='min')

    # Train the model
    model.fit(X_train, Y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1) # callbacks=[checkpoint]

    model.save_weights('data/rnn_model_weights.h5')

    return model


def load_variables(data_filename, model_weights, input_tok_name, target_tok_name):

    # Get the data
    anki = np.load(data_filename)

    # Get the tokenizers objects
    input_tokenizer, target_tokenizer = load_tokenizers(input_tok_name, target_tok_name)

    # Get the size of the vocabularies' sizes
    input_vocab_size = len(input_tokenizer.word_index) + 1   # Add one for <PAD> token
    target_vocab_size = len(target_tokenizer.word_index) + 1 # Add one for <PAD> token

    # Get maximim length of phrases to standardize all inputs to the same length (padding)
    input_length = max([len(line.split()) for line in anki[:, INPUT_COLUMN]])
    target_length = max([len(line.split()) for line in anki[:, TARGET_COLUMN]])

    model = build_model(input_vocab_size, input_length, target_vocab_size, target_length)
    model.load_weights(model_weights)

    return (anki, input_length, target_length, input_tokenizer, target_tokenizer, model)


if __name__ == "__main__":

    # Get the data, remove punctuation
    anki = prep_anki(r"data/en2pt.txt", sample=None)

    # Make sample with custom sizes: at least 3 words, no more than 6
    mask = np.array([(len(a.split()) >= 3) & (len(a.split()) <= 6) for (a,b) in zip(anki[:,0], anki[:, 1])], dtype=bool)

    anki_sample = anki[mask, :]
    np.save("data/en2pt_3to6words_10k.npy", anki_sample)

    anki_sample = anki_sample[np.random.choice(anki_sample.shape[0], 10000, replace=False), :]

    # Creae the tokenizers objects
    input_tokenizer, target_tokenizer = prep_tokenizers(anki_sample, MAX_VOCAB_SIZE, OOV_TOKEN, name_modifier="tokenizer_3to6_10k")

    # Get maximim length of phrases to standardize all inputs to the same length (padding)
    input_length = max([len(line.split()) for line in anki_sample[:, INPUT_COLUMN]])
    target_length = max([len(line.split()) for line in anki_sample[:, TARGET_COLUMN]])

    # Divide into train and test
    train, test = train_test_split(anki_sample, test_size=TEST_SIZE)

    # Define and train the model
    model = train_model(train, input_tokenizer, input_length, target_tokenizer, target_length)

    # Evaluate the model
    bleu_scores = evaluate_model(model, test, input_tokenizer, input_length, target_tokenizer, target_length, print_examples=True)
