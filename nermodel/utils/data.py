import sys

from .alphabet import Alphabet
from .functions import *

try:
    import cPickle as cPickle
except ImportError:
    import pickle as pickle

START = "</s>"
UNKOWN = "</unk>"
PADDING = "</pad>"


class Data:

    def __init__(self):
        self.MAX_SENTENCE_LENTGH = 1000
        self.MAX_WORD_LENGTH = -1
        self.number_normalized = False
        self.norm_word_emb = False
        self.norm_char_emb = False
        self.word_alphabet = Alphabet('word')
        self.char_alphabet = Alphabet('character')

        self.feature_name = []
        self.feature_alphabets = []
        self.feature_num = len(self.feature_alphabets)
        self.feat_config = None 
        

        self.label_alphabet = Alphabet('label', True)
        self.tagScheme = 'NoSeg'  # BMES/BIO (remplacer par du BIO)
        
        self.seg = True

        # IO
        self.train_dir = None
        self.dev_dir = None
        self.test_dir = None
        self.raw_dir = None

        self.decode_dir = None
        self.dset_dir = None  # data vocabulary related file
        self.model_dir = None # model save file
        self.load_model_dir = None  # model load file

        self.word_emb_dir = None
        self.char_emb_dir = None
        self.feature_emb_dirs = []

        self.train_texts = []
        self.dev_texts = []
        self.test_texts = []
        self.raw_texts = []

        self.train_Ids = []
        self.dev_Ids = []
        self.test_Ids = []
        self.raw_Ids = []

        self.pretrain_word_embedding = None
        self.pretrain_char_embedding = None
        self.pretrain_feature_embeddings = []

        self.label_size = 0
        self.word_alphabet_size = 0
        self.char_alphabet_size = 0
        self.label_alphabet_size = 0
        self.feature_emb_dims = []
        self.norm_feature_embs = []
        self.word_emb_dim = 50
        self.char_emb_dim = 30

        # Networks
        self.word_feature_extractor = "LSTM"  # LSTM/CNN/GRU
        self.use_char = True
        self.char_feature_extractor = "CNN"  # LSTM/**CNN**/GRU
        self.use_crf = True
        self.nbest = None

        # Training (not necessary for deployment but it could be useful if we
        # want to re-train models with an api)
        self.average_batch_loss = False
        self.optimizer = "SGD"
        self.status = "train"
        # Hyperparameters
        self.HP_cnn_layer = 4
        self.iteration = 100
        self.batch_size = 10
        self.HP_char_hidden_dim = 50
        self.HP_hidden_dim = 200
        self.HP_dropout = 0.5
        self.HP_lstm_layer = 1
        self.HP_bilstm = True

        self.HP_gpu = False
        self.HP_lr = 0.015
        self.HP_lr_decay = 0.05
        self.HP_clip = None
        self.HP_momentum = 0
        self.HP_l2 = 1e-8

    def show_data_summary(self):
        print('**' * 20)
        print("Data summary:")
        print(" Tag Scheme: {}".format(self.tagScheme))
        print(" Max sentence length: {}".format(self.MAX_SENTENCE_LENTGH))
        print(" Number normalized: {}".format(self.number_normalized))
        print(" Word Alphabet size: {}".format(self.word_alphabet_size))
        print(" Char Alphabet size: {}".format(self.char_alphabet_size))
        print(" Label Alphabet size: {}".format(self.label_alphabet_size))
        # to be continued

    # read_isntances
    def generate_instance(self, name):
        self.fix_alphabet()
        if name == 'train':
            self.train_texts, self.train_Ids = read_instance(
                self.train_dir, self.word_alphabet, self.char_alphabet, self.label_alphabet, self.feature_alphabets, self.number_normalized, self.MAX_SENTENCE_LENTGH)
        elif name == 'dev':
            self.dev_texts, self.dev_Ids = read_instance(
                self.dev_dir, self.word_alphabet, self.char_alphabet, self.label_alphabet, self.feature_alphabets, self.number_normalized, self.MAX_SENTENCE_LENTGH)
        elif name == 'test':
            self.test_texts, self.test_Ids = read_instance(
                self.test_dir, self.word_alphabet, self.char_alphabet, self.label_alphabet, self.feature_alphabets, self.number_normalized, self.MAX_SENTENCE_LENTGH)
        elif name == 'raw':
            self.raw_texts, self.raw_Ids = read_instance(
                self.raw_dir, self.word_alphabet, self.char_alphabet, self.label_alphabet, self.feature_alphabets, self.number_normalized, self.MAX_SENTENCE_LENTGH)
        else:
            print('Error: you can only generate from train/dev/test/raw')

    # allow to build instance form a list of tokens
    def generate_instance_from_list(self, input_data):
        self.fix_alphabet()
        # input in raw data
        self.raw_texts, self.raw_Ids = read_instance_from_list(input_data, self.word_alphabet, self.char_alphabet, self.label_alphabet, self.feature_alphabets, self.number_normalized, self.MAX_SENTENCE_LENTGH)

    def fix_alphabet(self):
        self.word_alphabet.close()
        self.char_alphabet.close()
        self.label_alphabet.close()
        
        for idx in range(self.feature_num):
            self.feature_alphabets[idx].close()
        

    # load dset file
    def load(self, data_file):
        f = open(data_file, 'rb')
        tmp_dict = pickle.load(f)
        f.close()
        self.__dict__.update(tmp_dict)
    # save dset file

    def save(self, save_file):
        f = open(save_file, 'wb')
        pickle.dump(self.__dict__, f, 2)
        f.close()

    # load export file: to be added in the train process
    def load_export(self, data_file):
        with open(data_file, 'rb') as f:
            tmp_dict = pickle.load(f)
        # re-create alphabets
        self.word_alphabet = Alphabet('word')
        self.char_alphabet = Alphabet('charcater')
        self.label_alphabet = Alphabet('label', True)

        self.word_alphabet.from_json(tmp_dict['word_alphabet'])
        self.char_alphabet.from_json(tmp_dict['char_alphabet'])
        self.label_alphabet.from_json(tmp_dict['label_alphabet'])
        
        tmp_dict['word_alphabet'] = self.word_alphabet
        tmp_dict['char_alphabet'] = self.char_alphabet
        tmp_dict['label_alphabet'] = self.label_alphabet

        self.__dict__.update(tmp_dict)

    # save export file
    def save_export(self, save_file):
        '''
           export an architecture specs and alphabets, erased other informations
        '''
        tmp_dict = self.__dict__
        exp_dict = {}
        # copy information in dset but with independance
        for k, v in tmp_dict.items():
            if k in 'word_alphabet':
                exp_dict[k] = Alphabet('word')
                exp_dict[k].from_json(v.get_content())
            elif k == 'char_alphabet':
                exp_dict[k] = Alphabet('char')
                exp_dict[k].from_json(v.get_content())
            elif k == 'label_alphabet':
                exp_dict[k] = Alphabet('label', True)
                exp_dict[k].from_json(v.get_content())
            else:
                exp_dict[k] = v
                
        # re-initialize un-useful informations (export should be light)
        exp_dict['word_alphabet'] = exp_dict['word_alphabet'].get_content()
        exp_dict['char_alphabet'] = exp_dict['char_alphabet'].get_content()
        exp_dict['label_alphabet'] = exp_dict['label_alphabet'].get_content()

        exp_dict['load_model_dir'] = None
        exp_dict['model_dir'] = None
        
        exp_dict['train_dir'] = None
        exp_dict['test_dir'] = None
        exp_dict['dev_dir'] = None
        exp_dict['raw_dir'] = None
        exp_dict['decode_dir'] = None
        exp_dict['word_emb_dir'] = None
        exp_dict['char_emb_dir'] = None

        exp_dict['train_texts'] = []
        exp_dict['test_texts'] = []
        exp_dict['dev_texts'] = []
        exp_dict['raw_texts'] = []

        exp_dict['train_Ids'] = []
        exp_dict['dev_Ids'] = []
        exp_dict['test_Ids'] = []
        exp_dict['raw_Ids'] = []

        exp_dict['pretrain_word_embedding'] = None
        exp_dict['pretrain_char_embedding'] = None

        with open(save_file, 'wb') as f:
            pickle.dump(exp_dict, f, 2)



    def read_config(self, config_file):
            # case we are reading a config file and not a python dictionnary
            # (config_file_to_dict not implementend yet)
        if isinstance(config_file, str):
            config = config_file_to_dict(config_file)
        else:
            config = config_file
        # read data:
        the_item = 'train_dir'
        if the_item in config:
            self.train_dir = config[the_item]
        the_item = 'dev_dir'
        if the_item in config:
            self.dev_dir = config[the_item]
        the_item = 'test_dir'
        if the_item in config:
            self.test_dir = config[the_item]
        the_item = 'raw_dir'
        if the_item in config:
            self.raw_dir = config[the_item]
        the_item = 'decode_dir'
        if the_item in config:
            self.decode_dir = config[the_item]
        the_item = 'dset_dir'
        if the_item in config:
            self.dset_dir = config[the_item]
        the_item = 'model_dir'
        if the_item in config:
            self.model_dir = config[the_item]
        the_item = 'load_model_dir'
        if the_item in config:
            self.load_model_dir = config[the_item]

        the_item = 'word_emb_dir'
        if the_item in config:
            self.word_emb_dir = config[the_item]
        the_item = 'char_emb_dir'
        if the_item in config:
            self.char_emb_dir = config[the_item]

        the_item = 'MAX_SENTENCE_LENGTH'
        if the_item in config:
            self.MAX_SENTENCE_LENGTH = int(config[the_item])
        the_item = 'MAX_WORD_LENGTH'
        if the_item in config:
            self.MAX_WORD_LENGTH = int(config[the_item])

        the_item = 'norm_word_emb'
        if the_item in config:
            self.norm_word_emb = str2bool(config[the_item])
        the_item = 'norm_char_emb'
        if the_item in config:
            self.norm_char_emb = str2bool(config[the_item])
        the_item = 'number_normalized'
        if the_item in config:
            self.number_normalized = str2bool(config[the_item])

        the_item = 'seg'
        if the_item in config:
            self.seg = str2bool(config[the_item])
        the_item = 'word_emb_dim'
        if the_item in config:
            self.word_emb_dim = int(config[the_item])
        the_item = 'char_emb_dim'
        if the_item in config:
            self.char_emb_dim = int(config[the_item])

        # read network:
        the_item = 'use_crf'
        if the_item in config:
            self.use_crf = str2bool(config[the_item])
        the_item = 'use_char'
        if the_item in config:
            self.use_char = str2bool(config[the_item])
        the_item = 'word_seq_feature'
        if the_item in config:
            self.word_feature_extractor = config[the_item]
        the_item = 'char_seq_feature'
        if the_item in config:
            self.char_feature_extractor = config[the_item]
        the_item = 'nbest'
        if the_item in config:
            self.nbest = int(config[the_item])

        the_item = 'feature'
        if the_item in config:
            self.feat_config = config[the_item]  # feat_config is a dict

        # read training setting:
        the_item = 'optimizer'
        if the_item in config:
            self.optimizer = config[the_item]
        the_item = 'ave_batch_loss'
        if the_item in config:
            self.average_batch_loss = str2bool(config[the_item])
        the_item = 'status'
        if the_item in config:
            self.status = config[the_item]

        # read Hyperparameters:
        the_item = 'cnn_layer'
        if the_item in config:
            self.HP_cnn_layer = int(config[the_item])
        the_item = 'iteration'
        if the_item in config:
            self.HP_iteration = int(config[the_item])
        the_item = 'batch_size'
        if the_item in config:
            self.HP_batch_size = int(config[the_item])

        the_item = 'char_hidden_dim'
        if the_item in config:
            self.HP_char_hidden_dim = int(config[the_item])
        the_item = 'hidden_dim'
        if the_item in config:
            self.HP_hidden_dim = int(config[the_item])
        the_item = 'dropout'
        if the_item in config:
            self.HP_dropout = float(config[the_item])
        the_item = 'lstm_layer'
        if the_item in config:
            self.HP_lstm_layer = int(config[the_item])
        the_item = 'bilstm'
        if the_item in config:
            self.HP_bilstm = str2bool(config[the_item])

        the_item = 'gpu'
        if the_item in config:
            self.HP_gpu = str2bool(config[the_item])
        the_item = 'learning_rate'
        if the_item in config:
            self.HP_lr = float(config[the_item])
        the_item = 'lr_decay'
        if the_item in config:
            self.HP_lr_decay = float(config[the_item])
        the_item = 'clip'
        if the_item in config:
            self.HP_clip = float(config[the_item])
        the_item = 'momentum'
        if the_item in config:
            self.HP_momentum = float(config[the_item])
        the_item = 'l2'
        if the_item in config:
            self.HP_l2 = float(config[the_item])


def str2bool(string):
    if string == 'True' or string == 'true' or string == 'TRUE':
        return True
    else:
        return False
