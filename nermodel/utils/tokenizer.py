import spacy
import re

EXCLUDE = ['^[ ]{1,}$', '\n\n*','\t\t*']

class myTokenizer():
    ''' Tokenizer
    '''

    def __init__(self, language ='en'):
        self.exclude = EXCLUDE
        self.language = language
        nlp = spacy.load(self.language, disable=['tagger', 'ner', 'parser'])
        nlp.add_pipe(nlp.create_pipe('sentencizer')) 
        self.nlp = nlp

    def tokenize(self, text):
        ''' Customization of the spaCy tokenizer
        input:
            text, raw text to tokenize
        output:
            list of sentences, as lists of words (nb_sent, sent_length)
        '''

        doc = self.nlp(text)
        sent_list = []
        sentence = []
        for sent in doc.sents:
            for token in sent:
                #print(token)
                add = True
                for ex in self.exclude:
                    if re.search(ex, token.string) is not None:
                        add = False
                if add:
                    sentence.append(token)
            if len(sentence) >= 1:
                sent_list.append(sentence)
            sentence = []
        return(sent_list)
