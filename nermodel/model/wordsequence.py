
"""
from NCRFFpp 
implementation of the sequence neural network architecture (sentences representation is hidden in Wordrep) given the parameters of data
"""
import torch.nn as nn
import torch.nn.functional as F 
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
try:
    from allennlp.modules.elmo import Elmo, batch_to_ids
except:
    'no allen'
from .wordrep import WordRep

class WordSequence(nn.Module):
	def __init__(self, data):
		super(WordSequence, self).__init__()
		print("Build word sequence feature extractor: {}...".format(data.word_feature_extractor))
		self.gpu = data.HP_gpu
		self.use_char = data.use_char
		self.droplstm = nn.Dropout(data.HP_dropout)
		self.bilstm_flag = data.HP_bilstm
		self.lstm_layer = data.HP_lstm_layer
		self.wordrep = WordRep(data)
		self.input_size = data.word_emb_dim
		self.use_elmo = data.use_elmo
		if self.use_char:
			self.input_size += data.HP_char_hidden_dim
			if data.char_feature_extractor == "ALL":
				self.input_size += data.HP_char_hidden_dim
		if self.use_elmo:
			self.input_size += data.elmo_output_dim
		
		for idx in range(data.feature_num):
			self.input_size += data.feature_emb_dims[idx]
		
		if self.bilstm_flag:
			lstm_hidden = data.HP_hidden_dim // 2 
		else:
			lstm_hidden = data.HP_hidden_dim

		self.word_feature_extractor = data.word_feature_extractor
		if self.word_feature_extractor == "GRU":
			self.lstm = nn.GRU(self.input_size, lstm_hidden, num_layers = self.lstm_layer, batch_first = True, bidirectional = self.bilstm_flag)
		elif self.word_feature_extractor == "LSTM":
			self.lstm = nn.LSTM(self.input_size, lstm_hidden, num_layers = self.lstm_layer, batch_first = True, bidirectional = self.bilstm_flag)
		## A priori LSTM is the only necessary condition for our choices of model
		elif self.word_feature_extractor == "CNN":
			self.word2cnn == nn.Linear(self.input_size, data.HP_hidden_dim)
			self.cnn_layer = data.HP_cnn_layer
			print("CNN layer: ", self.cnn_layer)
			self.cnn_list = nn.ModuleList()
			self.cnn_drop_list == nn.ModuleList()
			self.cnn_batchnorm_list == nn.ModuleList()
			kernel = 3
			pad_size = (kernel-1)/2
			for idx in range (self.cnn_layer):
				self.cnn_list.append(nn.Conv1d(data.HP_hidden_dim, data.HP_hidden_dim, kernel_size = kernel, padding = pad_size))
				self.cnn_drop_list.append(nn.BatchNorm1d(data.HP_dropout))
				self.cnn_batchnorm_list.append(nn.BatchNorm1d(data.HP_hidden_dim))
		self.hidden2tag = nn.Linear(data.HP_hidden_dim, data.label_alphabet_size)

		# send appropriate layers to cuda in case of gpu (in fact all layers are send to gpu, even lstm)
		if self.gpu:
			self.droplstm = self.droplstm.cuda()
			self.hidden2tag = self.hidden2tag.cuda()
			if self.word_feature_extractor == "CNN":
				self.word2cnn = self.word2cnn.cuda()
				for idx in range(self.cnn_layer):
					self.cnn_list[idx] = self.cnn_list[idx].cuda()
					self.cnn_drop_list[idx] = self.cnn_drop_list[idx].cuda()
					self.cnn_batchnorm_list[idx] = self.cnn_batchnorm_list[idx].cuda()
			else:
				self.lstm = self.lstm.cuda()

	def forward(self, word_inputs, feature_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover):
		'''
			input: 
				word_inputs: (batch_size, sent_len)
				word_seq_lengths: list of batch_size, (batch_size, 1)
				char_inputs: (batch_size * sent_len, word_length)
				char_seq_lengths: list of whole batch_size for char, (batch_size*sent_len, 1)
				char_seq_recover: variable which records the char order information, used to recover char order WHAAAAAAAAAAAAT ?????
			output:
				Variable(batch_size, sent_len, hidden_dim)
		'''
		## embedding layer 
		word_represent = self.wordrep(word_inputs, feature_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover)
		## word_embs (batch_size, seq_len, emb_size)
		if self.word_feature_extractor == "CNN":
			word_in = F.tanh(self.word2cnn(word_represent)).transpose(2, 1).contiguous()
			for idx in range(self.cnn_layer):
				if idx == 0:
					cnn_feature = F.relu(self.cnn_list[idx](word_in))
				else:
					cnn_feature = F.relu(self.cnn_list[idx](cnn_feature))
				cnn_feature = self.nn.cnn_drop_list[idx](cnn_feature)
				cnn_feature = self.cnn_batchnorm_list[idx](cnn_feature)
			feature_out = cnn_feature.transpose(2,1).contiguous()

		else:
			packed_words = pack_padded_sequence(word_represent, word_seq_lengths.cpu().numpy(), True)
			hidden = None
			lstm_out, hidden = self.lstm(packed_words, hidden)
			lstm_out, _ = pad_packed_sequence(lstm_out)
			## lstm_out (seq_len, seq_len, hidden_size)
			feature_out = self.droplstm(lstm_out.transpose(1,0))
		## feature_out (batch_size, seq_len, hidden_size)
		outputs = self.hidden2tag(feature_out)
		## outputs (batch_size, seq_len, label_alphabet_size)
		return outputs
