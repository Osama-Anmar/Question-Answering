import re
import numpy as np
def input_output_split(data, text_normalization1, text_normalization2, text_normalization3): 
    data_1 = [] 
    data_2 = [] 
    data_3 = []
    data_pairs = [sentences.split("<sep>") for sentences in data] 
    for i in range(len(data_pairs)):
        data_1.append(data_pairs[i][0])
        data_2.append(data_pairs[i][1])
        data_3.append(data_pairs[i][2])
    data_1 = [re.sub("'", '', text) for text in data_1]
    data_1 = list(map(text_normalization1, data_1))
    data_2 = list(map(text_normalization2, data_2))
    data_3 = list(map(text_normalization3, data_3))
    data_3 = ["<start>" + " "+line + " " +"<end>" for line in data_3]
    return data_1, data_2, data_3


def decoder_input_output(output_data):
    decoder_input = [sentences[:-1] for sentences in output_data] 
    decoder_output = [sentences[1:] for sentences in output_data]
    return decoder_input, decoder_output