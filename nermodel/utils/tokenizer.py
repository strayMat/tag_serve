import regex
import spacy
from spacy.pipeline import SentenceSegmenter

try:
    from spacy.lang.en import English
except:
    'no english language'
try:
    from spacy.lang.fr import French
except:
    'no french language'
    
EXCLUDE = ['^[ ]{1,}$', '\t\t*', '\n\n*']
# see https://spacy.io/usage/linguistic-features#section-tokenization  for tokenization and sentence segmentation improvements.

class myTokenizer():
    ''' Tokenizer class
    '''

    def __init__(self, language ='en'):
        self.exclude = EXCLUDE
        self.language = language
        if language == 'fr':
            nlp = French()
        else:
            nlp = English()
        #nlp.add_pipe(nlp.create_pipe('sentencizer')) 
        sbd = SentenceSegmenter(nlp.vocab, strategy=split_sents)
        nlp.add_pipe(sbd)
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
                add = True
                for ex in self.exclude:
                    if regex.search(ex, token.string) is not None:
                        add = False
                if add:
                    sentence.append(token)
            if len(sentence) >= 1:
                sent_list.append(sentence)
            sentence = []
        return(sent_list)

    
sent_boundaries = '|'.join(['\.', '\!', '\?'])
def split_sents(doc):
    ''' Custom sentence segmenter
    '''
    start = 0
    seen_newline = False
    for i, word in enumerate(doc):
        # rule0: check if current character is breakline and following is an upper case
        if (i+1) < len(doc):
            following = doc[i+1].text
        #custom_rules = (regex.search('\n\n*', word.text) is not None) & (regex.search('\p{Lu}', following) is not None)
        if seen_newline and not word.is_space:
            yield doc[start:word.i]
            start = word.i
            seen_newline = False
        elif (regex.search(sent_boundaries, word.text) is not None):
            seen_newline = True
        prec = word.text
    if start < len(doc):
        yield doc[start:len(doc)]
