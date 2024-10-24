import re

def delete_punctuations(text):
        Punctuation  = '''`”؛،¿¡۔，"\“'()-./:;?[]^_`{}'''
        for punctuation in Punctuation:
                text = text.replace(punctuation, ' ')
        return text


def arabic_text_normalization(text):
        text = re.sub('\u200d', '', text)
        text = delete_punctuations(text)
        text = re.sub(r'([,!?;:().-؟])', r' \1 ', text)
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(' +', ' ', text)
        text = re.sub(r'\n','', text)
        text = re.sub("                ", "",text)
        text = re.sub("    ", "", text)
        text = text.strip()
        return text
