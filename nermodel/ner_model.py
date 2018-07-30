#from __future__ import absolute_import
### Main functions to wrap the ner model 

import gc
import time
import random
import sys
import json
import torch
import torch.autograd as autograd
import torch.optim as optim
import numpy as np

try:
    from model.seqmodel import SeqModel
except:
    from .model.seqmodel import SeqModel

try:
    from utils.data import Data
    from utils.metric import get_ner_fmeasure
except:
    from .utils.data import Data
    from .utils.metric import get_ner_fmeasure


def data_initialization(data):
    if data.use_feats:
        data.initial_feature_alphabets()
    data.build_alphabet(data.train_dir)
    data.build_alphabet(data.dev_dir)
    data.build_alphabet(data.test_dir)
    data.fix_alphabet()


def lr_decay(optimizer, epoch, decay_rate, init_lr):
    ''' For SGD
    '''
    lr = init_lr/(1+decay_rate*epoch)
    print(" Learning rate is setted as: {}".format(lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer


# maybe not useful
def predict_check(pred_variable, gold_variable, mask_variable):
    ''' Accuracy calculation based on mask (which are token and which are paddings)
        input:
            pred_variable (batch_size, sent_len): pred tag result variable (torch Variable)
            gold_variable (batch, sent_len): gol result variable
            mask_variable (batch_size, sent_len): mask_variable
    '''
    pred = pred_variable.cpu().data.numpy()
    gold = gold_variable.cpu().data.numpy()
    mask = mask_variable.cpu().data.numpy()
    overlaped = (pred == gold)
    right_token = np.sum(overlaped * mask)
    total_token = mask.sum()

    return right_token, total_token


def load_model_decode(data, name, label_flag=True):
    print("Load Model from file", data.model_dir)
    model = SeqModel(data)
    ## handle GPU/non GPU issues
    map_location = lambda storage, loc: storage
    if data.HP_gpu:
        map_location = None
    ## load weights 
    model.load_state_dict(torch.load(data.load_model_dir, map_location = map_location))
    start_time = time.time()
    speed, acc, p, r, f, pred_results, pred_scores = evaluate(data, model, name, nbest=None, label_flag=label_flag)
    end_time = time.time()
    time_cost = end_time - start_time
    # distinguish between non-segmentation tasks (POS, CCG ) and segmentation tasks (word segmentation, ner, chuncking) for which f1 score is necessary
    if data.seg: 
        print("{}: time{:.2f}s, speed: {:.2f}st/s; acc: {:.4f}, p: {:.4f}, r: {:.4f}, f: {:.4f}".format(name, time_cost, speed, acc, p, r , f))
    else:
        print("{}: time{:.2f}s, speed: {:.2f}st/s; acc: {:.4f}".format(name, time_cost, speed, acc))
    # pred_scores is empty (it is only filled when nbest is not None)
    return pred_results, pred_scores


def build_model(data):
    ''' For deployment: instantiate the model based on data object architecture specifications
    '''
    print("Load Model weights from file", data.load_model_dir)
    start_time = time.time()
    model = SeqModel(data)
    ## handle GPU/non GPU issues
    map_location = lambda storage, loc: storage
    if data.HP_gpu:
        map_location = None
    # laoding the weights of the model from load_model_dir
    model.load_state_dict(torch.load(data.load_model_dir, map_location = map_location))
    
    end_time = time.time()
    time_cost = end_time - start_time
    return model

#### volatile_flag and autograd.variable seem to be pytroch 3.1, maybe in a second pass, change all in pytorch 4.0
## WHAT is mask, batch_wordrecover ??? to be understood
def batchify_with_label(input_batch_list, gpu, volatile_flag=False, label_flag=True):
    """
        input: list of words, chars and labels, various length. [[words, chars, labels], [words, chars, labels],...] (watch out could also be with features in 2d pos)
            words: word ids for one sentence, (batch_size, sent_len)
            chars: char ids for one sentence, various lengths, (batch_size, sent_len, each_word_length)
            label_flag, boolean to indicate if input_batch_list contains labels (could be handled with a test on the length of input_batch_list[0]),
            if False then input_batch_list should be [[words, chars], [words, chars], ...]
        output:
            zero padding for word and char with their batch length
            word_seq_tensor: (batch_size , max_sent_len) Variable
            word_seq_lengths: (batch_size,  1) Tensor
            char_seq_tensor: (batch_size*max_sent_len, max_word_len) Variable
            char_seq_lengths: (batch_size*max_sent_len, 1) Tensor
            char_seq_recover: (batch_size*max_sent_len, 1) recover char sequence order
            mask: (batch_size, max_sent_len)

    """

    batch_size = len(input_batch_list)
    words = [sent[0] for sent in input_batch_list]
    chars = [sent[1] for sent in input_batch_list]
    
    if label_flag:
        labels = [sent[-1] for sent in input_batch_list]
    ## watchout, if adding features, have to change the code of utils.functions.read_instances
    features = [np.asarray(sent[2]) for sent in input_batch_list]
    feature_num = len(features[0][0])
    #feature_num = 0

    word_seq_lengths = torch.LongTensor(list(map(len, words)))
    max_seq_len = word_seq_lengths.max()
    word_seq_tensor = autograd.Variable(torch.zeros((batch_size, max_seq_len)), volatile = volatile_flag).long()
    ## change all variable as the following lines to upgrade to torch 4.0 (not sure that all will be working but this is a beginning for the shiftto torch 4.0)
    #word_seq_tensor = autograd.Variable(torch.zeros((batch_size, max_seq_len)), requires_grad = volatile_flag).long()
    
    ## to be returned anyway
    label_seq_tensor = []
    if label_flag:
        label_seq_tensor = autograd.Variable(torch.zeros((batch_size, max_seq_len)), volatile = volatile_flag).long()
    
    feature_seq_tensors = []
    for idx in range(feature_num):
        feature_seq_tensors.append(autograd.Variable(torch.zeros((batch_size, max_seq_len)), volatile = volatile_flag).long())
    
    ## WHY byte TYPE ?????????? AND WHAT IS THAT MASK
    mask = autograd.Variable(torch.zeros((batch_size, max_seq_len)), volatile = volatile_flag).byte()
    if label_flag:
        for idx, (seq, label, seqlen) in enumerate(zip(words, labels, word_seq_lengths)):
            word_seq_tensor[idx, :seqlen] = torch.LongTensor(seq)
            label_seq_tensor[idx, :seqlen] = torch.LongTensor(label)
            mask[idx, :seqlen] = torch.Tensor([1]*seqlen)
            
            for idy in range(feature_num):
                feature_seq_tensors[idy][idx, :seqlen] = torch.LongTensor(features[idx][:, idy])
            
    else:
        for idx, (seq, seqlen) in enumerate(zip(words, word_seq_lengths)):
            word_seq_tensor[idx, :seqlen] = torch.LongTensor(seq)
            mask[idx, :seqlen] = torch.Tensor([1]*seqlen)
            
            for idy in range(feature_num):
                feature_seq_tensors[idy][idx, :seqlen] = torch.LongTensor(features[idx][:, idy])
            
    word_seq_lengths, word_perm_idx = word_seq_lengths.sort(0, descending = True)
    word_seq_tensor = word_seq_tensor[word_perm_idx]
    
    for idx in range(feature_num):
        feature_seq_tensors[idx] = feature_seq_tensors[idx][word_perm_idx]
    

    if label_flag:
        label_seq_tensor = label_seq_tensor[word_perm_idx]
    mask = mask[word_perm_idx]
    
    ## deal with char (padding)
    # pad_chars (batch_size, max_seq_len)
    pad_chars = [chars[idx] + [[0]] * (max_seq_len - len(chars[idx])) for idx in range(len(chars))] 
    length_list = [list(map(len, pad_char)) for pad_char in pad_chars]
    max_word_len = max(map(max, length_list))
    #print(batch_size, max_seq_len, max_word_len)
    char_seq_tensor = autograd.Variable(torch.zeros((batch_size, max_seq_len, max_word_len)), volatile = volatile_flag).long()
    char_seq_lengths = torch.LongTensor(length_list)
    for idx, (seq, seqlen) in enumerate(zip(pad_chars, char_seq_lengths)):
        for idy, (word, wordlen) in enumerate(zip(seq, seqlen)):
            char_seq_tensor[idx, idy, :wordlen] = torch.LongTensor(word)

    char_seq_tensor = char_seq_tensor[word_perm_idx].view(batch_size*max_seq_len, -1)
    char_seq_lengths = char_seq_lengths[word_perm_idx].view(batch_size*max_seq_len,)
    char_seq_lengths, char_perm_idx = char_seq_lengths.sort(0, descending = True) 
    char_seq_tensor = char_seq_tensor[char_perm_idx]
    _, char_seq_recover = char_perm_idx.sort(0, descending = False)
    _, word_seq_recover = word_perm_idx.sort(0, descending = False)

    if gpu:
        word_seq_tensor = word_seq_tensor.cuda()
        for idx in range(feature_num):
            feature_seq_tensors[idx] = feature_seq_tensors[idx].cuda() 
        
        word_seq_lengths = word_seq_lengths.cuda()
        word_seq_recover = word_seq_recover.cuda()
        if label_flag:
            label_seq_tensor = label_seq_tensor.cuda()
        char_seq_tensor = char_seq_tensor.cuda()
        char_seq_recover = char_seq_recover.cuda()
        mask = mask.cuda()

    return word_seq_tensor, feature_seq_tensors, word_seq_lengths, word_seq_recover, char_seq_tensor, char_seq_lengths, char_seq_recover, label_seq_tensor, mask


def recover_label(pred_variable, gold_variable, mask_variable, label_alphabet, word_recover):
    """
        input:
            pred_variable (batch_size, sent_len): pred tag result
            gold_variable, (batch_size, sent_len): gold_result variable
            mask_variable (batch_size, sent_len): mask_variable
    """

    pred_variable = pred_variable[word_recover]
    gold_variable = gold_variable[word_recover]
    mask_variable = mask_variable[word_recover]
    batch_variable = pred_variable.size(0)
    seq_len = pred_variable.size(1)
    mask = mask_variable.cpu().data.numpy()
    pred_tag = pred_variable.cpu().data.numpy()
    gold_tag = gold_variable.cpu().data.numpy()
    batch_size = mask.shape[0]
    pred_label = []
    gold_label = []
    for idx in range(batch_size):
        pred = [label_alphabet.get_instance(pred_tag[idx][idy]) for idy in range(seq_len) if mask[idx][idy] != 0]
        gold = [label_alphabet.get_instance(gold_tag[idx][idy]) for idy in range(seq_len) if mask[idx][idy] != 0]
        assert(len(pred) == len(gold))
        pred_label.append(pred)
        gold_label.append(gold)
    return pred_label, gold_label


def recover_pred_label(pred_variable, mask_variable, label_alphabet, word_recover):
    """
        input:
            pred_variable (batch_size, sent_len): pred tag result
            mask_variable (batch_size, sent_len): mask_variable
    """

    pred_variable = pred_variable[word_recover]
    mask_variable = mask_variable[word_recover]
    batch_variable = pred_variable.size(0)
    seq_len = pred_variable.size(1)
    mask = mask_variable.cpu().data.numpy()
    pred_tag = pred_variable.cpu().data.numpy()
    batch_size = mask.shape[0]
    pred_label = []
    for idx in range(batch_size):
        pred = [label_alphabet.get_instance(pred_tag[idx][idy]) for idy in range(seq_len) if mask[idx][idy] != 0]
        pred_label.append(pred)
    return pred_label


def evaluate(data, model, name, nbest=None, label_flag=True):
    ''' Evaluation of the model on a test data (or raw data wo labels)
    '''
    if name == "train":
        instances = data.train_Ids
    elif name == "dev":
        instances = data.dev_Ids
    elif name == "test":
        instances = data.test_Ids
    elif name == "raw":
        instances = data.raw_Ids
    else:
        print("Error: wrong evaluate name,", name)
        exit(1)
    right_token = 0
    whole_token = 0
    nbest_pred_results = []
    pred_scores = []
    pred_results = []
    gold_results = []
    ## set model in eval mode
    model.eval()
    batch_size = data.batch_size
    start_time = time.time()
    train_num = len(instances)
    total_batch = train_num//batch_size+1
    for batch_id in range(total_batch):
        start = batch_id*batch_size
        end = (batch_id+1)*batch_size
        if end > train_num:
            end = train_num
        instance = instances[start:end]
        if not instance:
            continue
        batch_word, batch_features, batch_wordlen, batch_wordrecover, batch_char, batch_charlen, batch_charrecover, batch_label, mask = batchify_with_label(instance, data.HP_gpu, True, label_flag = label_flag)
        # here in ncrfpp code, there is a nbest condition that I wont code
        tag_seq = model(batch_word, batch_features, batch_wordlen, batch_char, batch_charlen, batch_charrecover, mask)
        if label_flag:
            pred_label, gold_label = recover_label(tag_seq, batch_label, mask, data.label_alphabet, batch_wordrecover)
            gold_results += gold_label
        else:
            pred_label = recover_pred_label(tag_seq, mask, data.label_alphabet, batch_wordrecover)
        
        pred_results += pred_label

    decode_time = time.time() - start_time
    speed = len(instances)/decode_time
    if label_flag:
        acc, p, r, f = get_ner_fmeasure(gold_results, pred_results, data.tagScheme)
    else:
        acc, p, r, f = (0,0,0,0)
    return speed, acc, p, r, f, pred_results, pred_scores


def train(data):
    ''' initialize model and train
    '''
    print('Training model...')
    data.show_data_summary()
    # save dset 
    save_data_name = data.model_dir + '.dset'
    data.save(save_data_name)
    # save exportable model architecture (for deployment)
    data.save_export(data.model_dir + '.xpt')
    model = SeqModel(data)
    if data.optimizer.lower() == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=data.HP_lr, momentum=data.HP_momentum, weight_decay=data.HP_l2)
    elif data.optimizer.lower() == 'adagrad':
        optimizer = optim.Adagrad(model.parameters(), lr=data.HP_lr, weight_decay=data.HP_l2)
    elif data.optimizer.lower() == 'adadelta':
        optimizer = optim.SGD(model.parameters(), lr=data.HP_lr, weight_decay=data.HP_l2)
    elif data.optimizer.lower() == 'rmsprop':
        optimizer = optim.RMSprop(model.parameters(), lr=data.HP_lr, weight_decay=data.HP_l2)
    elif data.optimizer.lower() == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=data.HP_lr, weight_decay=data.HP_l2)
    else:
        print('Optimizer illegal: {}'.format(data.optimizer))
        exit(1)
    best_dev = -10
    ## start training
    for idx in range(data.iteration):
        epoch_start = time.time()
        temp_start = epoch_start
        print("Epoch {}/{}".format(idx,data.iteration))
        if data.optimizer.lower() == 'sgd':
            optimizer = lr_decay(optimizer, idx, data.HP_lr_decay, data.HP_lr)
        instance_count = 0
        sample_id = 0
        sample_loss = 0
        total_loss = 0 
        right_token = 0
        whole_token = 0
        random.shuffle(data.train_Ids)
        ## set model in train mode
        model.train()
        model.zero_grad()
        batch_size = data.batch_size
        #batch_id = 0
        train_num = len(data.train_Ids)
        total_batch = train_num // batch_size + 1
        for batch_id in range(total_batch):
            start = batch_id*batch_size
            end = (batch_id+1)*batch_size
            if end > train_num:
                end = train_num
            instance = data.train_Ids[start:end]
            if not instance:
                continue
            batch_word, batch_features, batch_wordlen, batch_wordrecover, batch_char, batch_charlen, batch_charrecover, batch_label, mask = batchify_with_label(instance, data.HP_gpu, volatile_flag = False, label_flag = True)
            #print(batch_char.size())
            #print(batch_char.max())
            instance_count += 1
            loss, tag_seq = model.neg_log_likelihood_loss(batch_word, batch_features, batch_wordlen, batch_char, batch_charlen, batch_charrecover, batch_label, mask)
            right, whole = predict_check(tag_seq, batch_label, mask)
            right_token += right
            whole_token += whole 
            sample_loss += loss.data[0]
            total_loss += loss.data[0]
            if end%500 == 0:
                temp_time = time.time()
                temp_cost = temp_time - temp_start
                temp_start = temp_time
                print(' Instance {}; Time {:.2}s; loss {:.4}; acc {}/{}={:.4}'.format(end, temp_cost, sample_loss, right_token, whole_token, (right_token + 0.)/whole_token))
                if sample_loss > 1e8 or str(sample_loss) == 'nan':
                    print('ERROR: LOSS EXPLOSION (>1e8) ! Please set adapted parameters and structure! EXIT ...')
                    exit(1)
                sys.stdout.flush()
                sample_loss = 0

            loss.backward()
            optimizer.step()
            model.zero_grad()

        temp_time = time.time()
        temp_cost = temp_time - temp_start
        print(' Instance {}; Time {:.2}s; loss {:.4}; acc {}/{}={:.4}'.format(end, temp_cost, sample_loss, right_token, whole_token, (right_token + 0.)/whole_token))

        epoch_finish = time.time()
        epoch_cost = epoch_finish - epoch_start
        print(' Epoch: {} training finished. Time: {:.2}s; speed: {:.2}st/s; total loss: {}'.format(idx, epoch_cost, train_num/epoch_cost, total_loss))
        if total_loss > 1e8 or str(sample_loss) == 'nan':
            print('ERROR: LOSS EXPLOSION (>1e8) ! Please set adapted parameters and structure! EXIT ...')
            exit(1)
        # continue
        speed, acc, p, r, f, _, _ = evaluate(data, model, "dev")
        dev_finish = time.time()
        dev_cost = dev_finish - epoch_finish
        # saving dev results json for model analysis
        dev_res = tuple((speed, acc, p, r, f))
        

        if data.seg:
            current_score = f
            print("Dev: time: {:.2}s, speed {:.2}st/s; acc: {:.4}, p: {:.4}, r: {:.4}, f: {:.4}".format(dev_cost, speed, acc, p, r, f))
        else:
            current_score = acc
            print("Dev: time: {:.2}s, speed {:.2}st/s; acc: {:.4}".format(dev_cost, speed, acc))

        # decode test
        speed, acc, p, r, f, _, _ = evaluate(data, model, 'test')
        test_finish = time.time()
        test_cost = test_finish - dev_finish
        if data.seg:
            print("Test: time: {:.2}s, speed {:.2}st/s; acc: {:.4}, p: {:.4}, r: {:.4}, f: {:.4}".format(dev_cost, speed, acc, p, r, f))
        else:
            print("Test: time: {:.2}s, speed {:.2}st/s; acc: {:.4}".format(dev_cost, speed, acc))

        if current_score > best_dev:
            if data.seg:
                print('"Exceed previous best f score:', best_dev)
            else:
                print('"Exceed previous best acc score:', best_dev)

            model_name = data.model_dir + '.' + str(idx) + '.model'
            print('Save current best model in file:', model_name)
            torch.save(model.state_dict(), model_name)
            best_dev = current_score
            path2info = data.model_dir + '.infos'
            save_infos(data, dev_res, path2info)

        gc.collect()
    print('Training done!')
    return best_dev

def save_infos(data, dev_res, path2info):
    ''' save informations of interest for model analysis
    '''
    HPOI = {
    'IO':['train_dir', 'dev_dir', 'test_dir','word_emb_dir'],
    
    'archi':['use_crf', 'HP_hidden_dim', 'HP_char_hidden_dim', 'HP_lstm_layer',
             'HP_bilstm', 'HP_cnn_layer', 'HP_dropout', 'word_emb_dim', 'char_emb_dim'],
    
    'train': ['batch_size', 'iteration', 'optimizer', 'HP_lr', 'HP_lr_decay', 'HP_momentum', 'HP_l2','HP_clip'],
    
    'others':['MAX_SENTENCE_LENTGH', 'HP_gpu', 'number_normalized']
    }

    # input parameters in infos dict
    infos = {}
    res_dict = data.__dict__
    for k, v in HPOI.items():
        infos[k] = {}
        for param in v:
            infos[k][param] = res_dict[param]
    # input dev results in infos dict
    infos['dev_res'] = {}
    infos['dev_res']['speed'] = dev_res[0]
    infos['dev_res']['acc'] = dev_res[1]
    infos['dev_res']['precision'] = dev_res[2]
    infos['dev_res']['recall'] = dev_res[3]
    infos['dev_res']['f1'] = dev_res[4]
    with open(path2info, 'w') as f:
        json.dump(infos, f)

    print('Save informations about model in file: {}'.format(path2info))
