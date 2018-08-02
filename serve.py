#from model import NERMODEL
#from model.config import Config
import time
import torch
import re
from nermodel.utils.data import Data
from nermodel.ner_model import build_model, evaluate

VERBOSE = True

def get_model_api(path2xpt, path2model):
    '''Returns lambda function for API'''
    # 1. Initialize model
    load_start = time.time()
    decode_config_dict = {'load_model_dir':path2model, # model weights
                            'xpt_dir':path2xpt}

    data = Data()
    print('**' * 30)
    print("NER MODEL: loading model, decoding-style ...")
    ## Loading model architecture parameters and path to the weights
    data.load_export(path2xpt)
    data.read_config(decode_config_dict)
    data.HP_gpu = torch.cuda.is_available()
    if VERBOSE:
        data.show_data_summary(deploy = True)
    
    ## building model and inputing the weights
    model = build_model(data)
    load_end = time.time() - load_start
    print("Model loaded in {:.2}s".format(load_end))
    print('**' * 30)

    def model_api(input_data, data = data, model = model, live = True, tokenizer = None):
        # 2. Make prediction given an input_data
        if tokenizer == None:
            print('EMPTY TOKENIZER! please specify a tokenizer...')
            exit(1)
        
        ## Pre-processing from client, very delicate (we have to keep the same tokenization for the model input and for the spans at the output
        text = tokenizer.tokenize(input_data)
        input_client = []
        input_model = []
        sentence = []
        for sent in text:
            for token in sent:
                w = token.string.strip()
                sentence.append(w)
            input_client += sentence
            input_model += sentence + ['']
            sentence = []
        
        start_time = time.time()
        #print(feed_data)
        data.generate_instance_from_list(input_model)
        #print('***************')
        #print(data.raw_texts)
        #print(evaluate(data, model, 'raw', label_flag=False))
        #print('*****************')
        speed, acc, p, r, f, pred_results, pred_scores = evaluate(data, model, 'raw', label_flag=False) 
        
        timed = time.time() - start_time
        print('Processing time {:.2} s'.format(timed))
        print('Decoding speed: {0:.2f} st/s'.format(speed))
        # reconstruct a unique sequence for the client
        if live: 
            output_client = []
            for l in pred_results:
                output_client += l
            output_aligned = align_data({'raw_input': input_client, 'labels':output_client})
            # return two aligned string sequences 
            return output_aligned['raw_input'], output_aligned['labels']
        else:
            # return the original text as well as a list of sentences where each sentence is a list of prediction (nb_sentences, sent_length)
            return input_data, pred_results
    return model_api


def align_data(data):
    ''' align a dictionnary of sequences 
        input:
            data, dict of sequences { 'key1': ['I', 'dream', 'of', 'the', 'Moon'] 
                        'key2': [O, O, O, O, 'B-LOC']}
        output:

            dict of strings {'key1': ['I dream of the Moon'] 
                             'key2': ['O O     O  O   B-LOC']}
    '''
    spacings = [max([len(seq[i]) for seq in data.values()]) for i in range(len(data[list(data.keys())[0]]))]

    data_aligned = dict()

    for key, seq in data.items():
        str_aligned = ""
        for token, spacing in zip(seq, spacings):
            str_aligned += token + " "*(spacing-len(token) + 1)
        data_aligned[key] = str_aligned
    return data_aligned
