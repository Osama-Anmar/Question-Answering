import numpy as np
import re

def index_the_words(Data, filter = None):
    if filter != None:
        pattern = filter
        Data = [re.sub(pattern, " ", text) for text in Data]
    words = set() 
    for text in Data:
        for word in text.split():
            words.add(word)

    all_words = len(words)
    words_to_index = {word:index+1 for index, word in enumerate(words)} 
    index_to_word = {index+1:word for index, word in enumerate(words)}  
    words_to_index['UNK'] = 0
    index_to_word[0] = "UNK"
    words_to_index = dict(sorted(words_to_index.items(), key=lambda item: item[1]))
    index_to_word = dict(sorted(index_to_word.items(), key=lambda item: item[0]))
    return all_words + 1  , words_to_index, index_to_word 

def word_counts(Data):
     
    word_count = dict()
    for sentence in Data:
        for word_ in sentence.split():
            if word_ in word_count:
                word_count[word_] += 1
            else:
                word_count[word_] = 1
        word_count = dict(sorted(word_count.items(), key=lambda item: item[1], reverse=True))
        return word_count

def text_to_sequence(word_index, text):
     sequence_text = [[word_index[word] if word in word_index else word_index["UNK"] for word in sentences.split()] for sentences in text]
     return sequence_text


def sequences_padding(padding = 'post', input_sequence = None, max_length = None, truncating = 'post'):
    for i in range(0, len(input_sequence)):
        while len(input_sequence[i]) < max_length:
            if padding == "post":
               input_sequence[i].insert(len(input_sequence[i]), 0)
            if padding == 'pre':
                input_sequence[i].insert(0, 0)
        if truncating == 'pre':
             input_sequence[i] =  input_sequence[i][-max_length:]
        if truncating == 'post':
             input_sequence[i] =  input_sequence[i][:max_length]
    return  np.array(input_sequence)

def one_hot_encoding(labels, all_words):
     label = np.zeros((len(labels), all_words))
     for i, j in enumerate(labels):
            label[i, j] = 1
     return label

def word_sequence_to_text(index_to_words, sequence, all_words):
     word_sequence_to_text_ = [index_to_words[index] if index in index_to_words else index_to_words[all_words] for index in sequence]
     return " ".join(word_sequence_to_text_)