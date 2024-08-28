import numpy as np
import re

class TextProcessing():
    def data_words(self, data, filter = None):
        def filter_data(pattern):
            translator = str.maketrans('', '', pattern) 
            filtered_data = [word.translate(translator) for word in data]
            return filtered_data 
        
        if filter:
            data = filter_data(filter)
        words = set() 
        for txt in data:
            for word in txt.split():
                words.add(word) 
        all_words = len(words)
        return all_words + 1, words
    

    def words_to_index_(self, words):
        words_to_index = {word:index+1 for index, word in enumerate(words)} 
        words_to_index['UNK'] = 0
        words_to_index = dict(sorted(words_to_index.items(), key=lambda item: item[1]))
        return words_to_index
    
    def index_to_word_(self, words):
        index_to_word = {index+1:word for index, word in enumerate(words)}  
        index_to_word[0] = "UNK"
        index_to_word = dict(sorted(index_to_word.items(), key=lambda item: item[0]))
        return index_to_word
    
    def word_counts(self, data):
        word_count = dict()
        for sentence in data:
            for word_ in sentence.split():
                if word_ in word_count:
                    word_count[word_] += 1
                else:
                    word_count[word_] = 1
        word_count = dict(sorted(word_count.items(), key=lambda item: item[1], reverse=True))
        return word_count
    
    def text_to_sequence(self, word_index, data):
     sequence_text = [[word_index[word] if word in word_index else word_index["UNK"] for word in sentences.split()] for sentences in data]
     return sequence_text
    
    def sequences_padding(self, padding = 'post', input_sequence = None, max_length = None, truncating = 'post'):
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
    
    def word_sequence_to_text(self, index_to_words, sequence):
     word_sequence_to_text_ = [index_to_words[index] if index in index_to_words else index_to_words["UNK"] for index in sequence]
     return " ".join(word_sequence_to_text_)