def data_to_list(data1, data2):
    question_list = []
    for line in data1:
        question_list.append(line)

    answer_list = []
    for line in data2:
        answer_list.append(line)
    return question_list, answer_list


def start_end_sentences(question, answer):
    question = ["<sos>"+ " " + line + " " +"<eos>" for line in question]
    answer = ["<sos>"+ " " + line + " " + "<eos>" for line in answer]
    return question, answer