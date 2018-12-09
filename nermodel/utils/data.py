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
        self.MAX_SENTENCE_LENGTH = 1000
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
        self.tagScheme = 'NoSeg'
        self.seg = True

        # IO
        self.train_dir = None
        self.dev_dir = None
        self.test_dir = None
        self.raw_dir = None
        self.elmo_opt = None
        self.elmo_weights = None

        self.decode_dir = None
        self.dset_dir = None  # data vocabulary related file
        self.xpt_dir = None # architecture and vocabulary (light replacement of dset)
        self.model_dir = None  # model save file
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
        
        # Network
        self.word_feature_extractor = "LSTM"  # LSTM/CNN/GRU
        self.use_char = True
        self.char_feature_extractor = "CNN"  # LSTM/**CNN**/GRU
        self.use_crf = True
        self.use_feats = False
        self.nbest = None
        self.use_elmo = False
        # Train
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

    def show_data_summary(self, deploy=False):
        print('**' * 20)
        print("----------Data summary:----------\n")

        print(" HP_gpu: {}".format(self.HP_gpu))
        print(" MAX_SENTENCE_LENGTH: {}".format(self.MAX_SENTENCE_LENGTH))
        print(" number_normalized: {}".format(self.number_normalized))
        print(" word_alphabet: {}".format(self.word_alphabet_size))
        print(" char_alphabet_size: {}".format(self.char_alphabet_size))
        print(" label_alphabet_size: {}".format(self.label_alphabet_size))
        print(" load_model_dir: {}".format(self.load_model_dir))

        if not deploy:
            print('\n')
            print('I/O:')
            print(" tagScheme: {}".format(self.tagScheme))
            print(" train_dir: {}".format(self.train_dir))
            print(" dev_dir: {}".format(self.dev_dir))
            print(" test_dir: {}".format(self.test_dir))
            print(" raw_dir: {}".format(self.raw_dir))
            print(" elmo_opt: {}".format(self.elmo_opt))
            print(" elmo_weights: {}".format(self.elmo_weights))
            print(" dset_dir: {}".format(self.dset_dir))
            print(" word_emb_dir: {}".format(self.word_emb_dir))
            print(" char_emb_dir: {}".format(self.char_emb_dir))
            print(" feature_emb_dirs: {}".format(self.feature_emb_dirs))


        print('\n')
        print('Network:')
        print(" word_feature_extractor: {}".format(self.word_feature_extractor))
        print(" use_char: {}".format(self.use_char))
        print(" char_feature_extractor: {}".format(self.char_feature_extractor))
        print(" use_crf: {}".format(self.use_crf))
        print(' use_elmo: {}'.format(self.use_elmo))
        print('\n')
        print('Network Hyperparameters:')
        print(" word_emb_dim: {}".format(self.word_emb_dim))
        print(" char_emb_dim: {}".format(self.char_emb_dim))
        print(" feature_emb_dims: {}".format(self.feature_emb_dims))
        print(" HP_char_hidden_dim: {}".format(self.HP_char_hidden_dim))
        print(" HP_hidden_dim: {}".format(self.HP_hidden_dim))
        print(" HP_lstm_layer: {}".format(self.HP_lstm_layer))
        print(" HP_bilstm: {}".format(self.HP_bilstm))
        print(" HP_cnn_layer: {}".format(self.HP_cnn_layer))
        print(" HP_dropout: {}".format(self.HP_dropout))

        if not deploy:
            print('\n')
            print('Training Hyperparameters:')
            print(" average_batch_loss: {}".format(self.average_batch_loss))
            print(" optimizer: {}".format(self.optimizer))
            print(" iteration: {}".format(self.iteration))
            print(" batch_size: {}".format(self.batch_size))
            print(" HP_lr: {}".format(self.HP_lr))
            print(" HP_lr_decayr: {}".format(self.HP_lr_decay))
            print(" HP_clip: {}".format(self.HP_clip))
            print(" HP_momentum: {}".format(self.HP_momentum))
            print(" HP_l2: {}".format(self.HP_l2))

        print('**' * 20 + '\n')

    # read_isntances
    def generate_instance(self, name):
        self.fix_alphabet()
        if name == 'train':
            self.train_texts, self.train_Ids = read_instance(
                self.train_dir, self.word_alphabet, self.char_alphabet, self.label_alphabet, self.feature_alphabets, self.number_normalized, self.MAX_SENTENCE_LENGTH)
        elif name == 'dev':
            self.dev_texts, self.dev_Ids = read_instance(
                self.dev_dir, self.word_alphabet, self.char_alphabet, self.label_alphabet, self.feature_alphabets, self.number_normalized, self.MAX_SENTENCE_LENGTH)
        elif name == 'test':
            self.test_texts, self.test_Ids = read_instance(
                self.test_dir, self.word_alphabet, self.char_alphabet, self.label_alphabet, self.feature_alphabets, self.number_normalized, self.MAX_SENTENCE_LENGTH)
        elif name == 'raw':
            self.raw_texts, self.raw_Ids = read_instance(
                self.raw_dir, self.word_alphabet, self.char_alphabet, self.label_alphabet, self.feature_alphabets, self.number_normalized, self.MAX_SENTENCE_LENGTH)
        else:
            print('Error: you can only generate from train/dev/test/raw')

    # allow to build instance form a list of tokens
    def generate_instance_from_list(self, input_data):
        self.fix_alphabet()
        # input in raw data
        self.raw_texts, self.raw_Ids = read_instance_from_list(
            input_data, self.word_alphabet, self.char_alphabet, self.label_alphabet, self.feature_alphabets, self.number_normalized, self.MAX_SENTENCE_LENGTH)

    def fix_alphabet(self):
        self.word_alphabet.close()
        self.char_alphabet.close()
        self.label_alphabet.close()

        for idx in range(self.feature_num):
            self.feature_alphabets[idx].close()

    def initial_feature_alphabets(self):
        items = open(self.train_dir, 'r').readline().strip('\n').split()
        total_column = len(items)
        if total_column > 2:
            for idx in range(1, total_column - 1):
                feature_prefix = items[idx].split(']', 1)[0] + ']'
                self.feature_alphabets.append(Alphabet(feature_prefix))
                self.feature_name.append(feature_prefix)
                print("Find feature: {}".format(feature_prefix))
        self.feature_num = len(self.feature_alphabets)
        self.pretrain_feature_embeddings = [None] * self.feature_num
        self.feature_emb_dims = [20] * self.feature_num
        self.feature_emb_dirs = [None] * self.feature_num
        self.norm_feature_embs = [False] * self.feature_num
        self.feature_alphabet_sizes = [0] * self.feature_num
        if self.feat_config:
            for idx in range(self.feature_num):
                if self.feature_name[idx] in self.feat_config:
                    self.feature_emb_dims[idx] = self.feat_config[
                        self.feature_name[idx]]['emb_size']
                    self.feature_emb_dirs[idx] = self.feat_config[
                        self.feature_name[idx]]['emb_dir']
                    self.norm_feature_embs[idx] = self.feat_config[
                        self.feature_name[idx]]['emb_norm']

    def build_alphabet(self, input_file):
        in_lines = open(input_file, 'r').readlines()
        for line in in_lines:
            if len(line) > 2:
                pairs = line.strip().split()
                word = pairs[0]
                if self.number_normalized:
                    word = normalize_word(word)
                label = pairs[-1]
                self.label_alphabet.add(label)
                self.word_alphabet.add(word)
                # build feautre alphabet
                for idx in range(self.feature_num):
                    feat_idx = pairs[idx + 1].split(']', 1)[-1]
                    self.feature_alphabets[idx].add(feat_idx)
                # build char alphabet
                for char in word:
                    self.char_alphabet.add(char)
        self.word_alphabet_size = self.word_alphabet.size()
        self.char_alphabet_size = self.char_alphabet.size()
        self.label_alphabet_size = self.label_alphabet.size()
        for idx in range(self.feature_num):
            self.feature_alphabet_sizes[
                idx] = self.feature_alphabets[idx].size()
        startS = False
        startB = False
        for label, _ in self.label_alphabet.iteritems():
            if "S-" in label.upper():
                startS = True
            elif "B-" in label.upper():
                startB = True
        if startB:
            if startS:
                self.tagScheme = "BMES"
            else:
                self.tagScheme = "BIO"

    def build_pretrain_emb(self):
        if self.word_emb_dir:
            print("Load pretrained word embedding, norm {}, dir: {}".format(
                self.norm_word_emb, self.word_emb_dir))
            self. pretrain_word_embedding, self.word_emb_dim = build_pretrain_embedding(
                self.word_emb_dir, self.word_alphabet, self.word_emb_dim, self.norm_word_emb)
        if self.char_emb_dir:
            print("Load pretrained char embedding, norm {}, dir: {}".format(
                self.norm_char_emb, self.char_emb_dir))
            self. pretrain_char_embedding, self.char_emb_dim = build_pretrain_embedding(
                self.char_emb_dir, self.char_alphabet, self.char_emb_dim, self.norm_char_emb)
        for idx in range(self.feature_num):
            if self.feature_emb_dirs[idx]:
                print("Load pretrained feature embedding, norm {}, dir: {}".format(
                    self.norm_feature_embs[idx], self.feature_emb_dirs[idx]))
                self. pretrain_feature_embeddings[idx], self.feature_emb_dims[idx] = build_pretrain_embedding(
                    self.feature_emb_dirs[idx], self.feature_alphabets[idx], self.feature_emb_dirs[idx], self.norm_feature_embs[idx])

    def write_decoded_results(self, predict_results, name):
        ''' Write predicted results in a given file (to the conll format)
        '''
        fout = open(self.decode_dir, 'w')
        sent_num = len(predict_results)
        content_list = []
        if name == 'raw':
            content_list = self.raw_texts
        elif name == 'test':
            content_list = self.test_texts
        elif name == 'dev':
            content_list = self.dev_texts
        elif name == 'train':
            content_list = self.train_texts
        else:
            print('Error: Please use a name in raw/train/dev/test ! ')
        assert(sent_num == len(content_list))
        for idx in range(sent_num):
            sent_length = len(predict_results[idx])
            for idy in range(sent_length):
                # content_list[idx] is a list of [word, char, features, label]
                # for python 2
                if sys.version_info[0] < 3:
                    fout.write(content_list[idx][0][idy].encode('utf-8') + ' ' + predict_results[idx][idy] + '\n')
                else:
                    fout.write(content_list[idx][0][idy] + ' ' + predict_results[idx][idy] + '\n')
            fout.write('\n')
        fout.close()
        print('Predict {} result has been written into file {}'.format(name, self.decode_dir))


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
        the_item = 'xpt_dir'
        if the_item in config:
            self.xpt_dir = config[the_item]
        the_item = 'model_dir'
        if the_item in config:
            self.model_dir = config[the_item]
        the_item = 'load_model_dir'
        if the_item in config:
            self.load_model_dir = config[the_item]
        the_item = 'elmo_opt'
        if the_item in config:
            self.elmo_opt = config[the_item]
        the_item = 'elmo_weights'
        if the_item in config:
            self.elmo_weights = config[the_item]


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
        the_item = 'use_feats'
        if the_item in config:
            self.use_feats == str2bool(config[the_item])
        the_item = 'word_feature_extractor'
        if the_item in config:
            self.word_feature_extractor = config[the_item]
        the_item = 'char_feature_extractor'
        if the_item in config:
            self.char_feature_extractor = config[the_item]
        the_item = 'nbest'
        if the_item in config:
            self.nbest = int(config[the_item])
        the_item = 'use_elmo'
        if the_item in config:
            self.use_elmo = config[the_item]

        the_item = 'feature'
        if the_item in config:
            self.feat_config = config[the_item]  # feat_config is a dict

        # read training setting:
        the_item = 'optimizer'
        if the_item in config:
            self.optimizer = config[the_item]
        the_item = 'average_batch_loss'
        if the_item in config:
            self.average_batch_loss = str2bool(config[the_item])
        the_item = 'status'
        if the_item in config:
            self.status = config[the_item]

        # read Hyperparameters:
        the_item = 'HP_cnn_layer'
        if the_item in config:
            self.HP_cnn_layer = int(config[the_item])
        the_item = 'iteration'
        if the_item in config:
            self.iteration = int(config[the_item])
        the_item = 'batch_size'
        if the_item in config:
            self.batch_size = int(config[the_item])

        the_item = 'HP_char_hidden_dim'
        if the_item in config:
            self.HP_char_hidden_dim = int(config[the_item])
        the_item = 'HP_hidden_dim'
        if the_item in config:
            self.HP_hidden_dim = int(config[the_item])
        the_item = 'HP_dropout'
        if the_item in config:
            self.HP_dropout = float(config[the_item])
        the_item = 'HP_lstm_layer'
        if the_item in config:
            self.HP_lstm_layer = int(config[the_item])
        the_item = 'HP_bilstm'
        if the_item in config:
            self.HP_bilstm = str2bool(config[the_item])

        the_item = 'HP_gpu'
        if the_item in config:
            self.HP_gpu = str2bool(config[the_item])
        the_item = 'HP_lr'
        if the_item in config:
            self.HP_lr = float(config[the_item])
        the_item = 'HP_lr_decay'
        if the_item in config:
            self.HP_lr_decay = float(config[the_item])
        the_item = 'HP_clip'
        if the_item in config:
            self.HP_clip = float(config[the_item])
        the_item = 'HP_momentum'
        if the_item in config:
            self.HP_momentum = float(config[the_item])
        the_item = 'HP_l2'
        if the_item in config:
            self.HP_l2 = float(config[the_item])
        

def str2bool(string):
    string = str(string)
    if string == 'True' or string == 'true' or string == 'TRUE':
        return True
    else:
        return False
