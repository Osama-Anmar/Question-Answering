import re
contractions = { 
"ain't": "am not , are not , is not , has not , have not",
"aren't": "are not ",
"can't": "can not",
"can't've": "can not have",
"'cause": "because",
"could've": "could have",
"couldn't": "could not",
"couldn't've": "could not have",
"didn't": "did not",
"doesn't": "does not",
"don't": "do not",
"hadn't": "had not",
"hadn't've": "had not have",
"hasn't": "has not",
"haven't": "have not",
"he'd": "he had , he would",
"he'd've": "he would have",
"he'll": "he shall , he will",
"he'll've": "he shall have , he will have",
"he's": "he has , he is",
"how'd": "how did",
"how'd'y": "how do you",
"how'll": "how will",
"how's": "how has , how is , how does",
"i'd": "i had , i would",
"i'd've": "i would have",
"i'll": "i shall , i will",
"i'll've": "i shall have , i will have",
"i'm": "i am",
"d'you": "do you",
"chang'd": "changed",
"i've": "i have",
"isn't": "is not",
"it'd": "it had , it would",
"it'd've": "it would have",
"it'll": "it shall , it will",
"it'll've": "it shall have , it will have",
"it's": "it has , it is",
"let's": "let us",
"'gainst": "against",
"sigh'd": "sighed",
"ma'am": "madam",
"mayn't": "may not",
"might've": "might have",
"mightn't": "might not",
"mightn't've": "might not have",
"must've": "must have",
"mustn't": "must not",
"mustn't've": "must not have",
"needn't": "need not",
"needn't've": "need not have",
"o'clock": "of the clock",
"oughtn't": "ought not",
"oughtn't've": "ought not have",
"shan't": "shall not",
"sha'n't": "shall not",
"shan't've": "shall not have",
"she'd": "she had , she would",
"she'd've": "she would have",
"she'll": "she shall , she will",
"she'll've": "she shall have , she will have",
"she's": "she has , she is",
"should've": "should have",
"shouldn't": "should not",
"shouldn't've": "should not have",
"ne'er": "never",
"so've": "so have",
"so's": "so as , so is",
"rul'd": "ruled",
"by'r ": "By our",
"i'faith": "in faith",
"return'd": "returned",
"fram'd": "framed",
"shriv'd": "shrived",
"gi' go-den": "give good evening",
"conceal'd": "concealed",
"spurn'd": "spurned",
"call'd": "called",
"lov'd": "loved",
"Achiev'd": "Achieved",
"pleas'd": "pleased",
"Ne'er": "Never",
"unpleasant'st": "most unpleasant",
"engag'd": "engaged",
"e'er": "ever",
"call'dst": "calledst",
"damn'd": "damned",
"mak'st": "makest",
"'tis": "it is",
"'twere": "it were",
"obtain'd": "obtained",
"that'd": "that would , that had",
"that'd've": "that would have",
"that's": "that has , that is",
"there'd": "there had , there would",
"there'd've": "there would have",
"there's": "there has , there is",
"they'd": "they had , they would",
"they'd've": "they would have",
"they'll": "they shall , they will",
"they'll've": "they shall have , they will have",
"they're": "they are",
"they've": "they have",
"to've": "to have",
"wasn't": "was not",
"we'd": "we had , we would",
"we'd've": "we would have",
"we'll": "we will",
"we'll've": "we will have",
"we're": "we are",
"we've": "we have",
"weren't": "were not",
"what'll": "what shall , what will",
"what'll've": "what shall have , what will have",
"what're": "what are",
"what's": "what has , what is",
"what've": "what have",
"when's": "when has , when is",
"when've": "when have",
"where'd": "where did",
"where's": "where has , where is",
"where've": "where have",
"who'll": "who shall , who will",
"who'll've": "who shall have , who will have",
"who's": "who has , who is",
"express'd": "expressed",
"arm'd": "armed",
"prepar'd": "prepared",
"refus'd": "refused",
"who've": "who have",
"why's": "why has , why is",
"deceiv'd":"deceived",
"why've": "why have",
"will've": "will have",
"for't": "for it",
"th'": "the",
"stay'd": "stayed",
"deserv'd": "deserved",
"amaz'd": "amazed",
"inter'gatory": "interrogatory",
"possess'd": "possessed",
"bechanc'd": "bechanced",
"won't": "will not",
"won't've": "will not have",
"would've": "would have",
"wouldn't": "would not",
"wouldn't've": "would not have",
"y'all": "you all",
"y'all'd": "you all would",
"y'all'd've": "you all would have",
"y'all're": "you all are",
"y'all've": "you all have",
"you'd": "you had , you would",
"you'd've": "you would have",
"you'll": "you shall , you will",
"you'll've": "you shall have , you will have",
"you're": "you are",
"you've": "you have",
}
contractions = dict(sorted(contractions.items(), key=lambda x: len(x[0]), reverse=True))


def contractions_(text):
    for contraction, expansion in contractions.items():
        text = text.replace(contraction, expansion)
    return text

def convert_number(text):
     text = re.sub('0', '٠', text)
     text = re.sub('1', '١', text)
     text = re.sub('2', '٢', text)
     text = re.sub('3', '٣', text)
     text = re.sub('4', '٤', text)
     text = re.sub('5', '٥', text)
     text = re.sub('6', '٦', text)
     text = re.sub('7', '٧', text)
     text = re.sub('8', '٨', text)
     text = re.sub('9', '٩', text)
     return text

def english_text_normalization(text):
        text = re.sub('\u200d', '', text)
        text = contractions_(text)
        text = convert_number(text)
        text = re.sub(r'([,!?;:().-])', r' \1 ', text)
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(' +', ' ', text)
        text = re.sub(r'\n','', text)
        text = re.sub("                ", "",text)
        text = re.sub("    ", "", text)
        text = text.strip()
        return text


def arabic_text_normalization(text):
        text = re.sub('\u200d', '', text)
        text = re.sub('\u200F', '', text)
        text = re.sub('\u200B', '', text)
        text = re.sub('\u200C', '', text)
        text = convert_number(text)
        text = re.sub(r'([,!؟;:().،-])', r' \1 ', text)
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(' +', ' ', text)
        text = re.sub(r'\n','', text)
        text = re.sub("                ", "",text)
        text = re.sub("    ", "", text)
        text = text.strip()
        return text
