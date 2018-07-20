import sys
import numpy as np 

def normalize_word(word):
	"normlize the digits to 0 (we could try some things more subtle (leave the dates or other things like thats"
	new_word = ""
	for char in word:
		if char.isdigit():
			new_word += '0'
		else:
			new_word += char
	return new_word

def read_instance(input_file, word_alphabet, char_alphabet, label_alphabet, feature_alphabets, number_normalized, max_sent_length, char_padding_size = -1, char_padding_symbol = '</pad>'):
	feature_num = len(feature_alphabets)
	with open(input_file, 'r') as f:
		in_lines = f.readlines()
	instance_texts = []
	instance_Ids = []
	words = []
	features = []
	chars = []
	labels = []
	word_Ids = []
	features_Ids = []
	char_Ids = []
	label_Ids = []
	for line in in_lines:
		# case there is something more than only a blank line 
		if len(line) > 2:
			# here it await for a space separated columns
			pairs = line.strip().split()

			if sys.version_info[0] < 3:
				word = pairs[0].decode('utf-8')
			else:
				word = pairs[0]

			if number_normalized:
				word = normalize_word(word)
			label = pairs[-1]
			words.append(word)
			labels.append(label)
			word_Ids.append(word_alphabet.get_index(word))
			label_Ids.append(label_alphabet.get_index(label))
			## get features
			feat_list = []
			feat_Id = []
			for idx in range(feature_num):
				feat_idx = pairs[idx+1].split(']', 1)[-1]
				feat_list.append(feat_idx)
				feat_Id.append(feature_alphabets[idx].get_index(feat_idx))
			features.append(feat_list)
			features_Ids.append(feat_Id)
			
			## get char
			char_list = []
			char_Id = []
			for char in word:
				char_list.append(char)
			# WHAT DONT UNDERSTAND THE ROLE OF PADDING CHAR (not really important, by default, padding is set to -1 but thought)
			if char_padding_size > 0:
				char_number = len(char_list)
				if char_number < char_padding_size:
					char_list = char_list + [char_padding_symbol] * (char_padding_size - char_number)
				assert(len(char_list) == char_padding_size)
			else:
				### not padding
				pass
			for char in char_list:
				char_Id.append(char_alphabet.get_index(char))
			# append lst of chars and list of char_ids to respective super_list (over the whole input_file)
			chars.append(char_list)
			char_Ids.append(char_Id)
		## case blankline (end of sentence / instance), here it sets aside the sentences too long (len >= max_sent_length)
		else:
			if (len(words) > 0) and ((max_sent_length < 0) or (len(words) < max_sent_length)):
				# suppressed features and feature_Ids from these lists
				instance_texts.append([words, chars, features, labels])
				instance_Ids.append([word_Ids, char_Ids, features_Ids, label_Ids])
			words = []
			features = []
			chars = []
			labels = []
			word_Ids = []
			features_Ids = []
			char_Ids = []
			label_Ids = []
	return instance_texts, instance_Ids

def read_instance_from_list(input_data, word_alphabet, char_alphabet, label_alphabet, feature_alphabets, number_normalized, max_sent_length, char_padding_size = -1, char_padding_symbol = '</pad>', data = ""):
	'''
		input:
			input_data, list of tokens where sentences are separeted by a a blank token '' ['word1', 'word2', 'word3', '', 'word4', ..]
	'''
	feature_num = len(feature_alphabets)
	instance_texts = []
	instance_Ids = []
	words = []
	features = []
	chars = []
	#labels = []
	word_Ids = []
	features_Ids = []
	char_Ids = []
	#label_Ids = []
	for line in input_data:
		
		#print(line + '\n')
		# case there is something more than only a blank line 
		if len(line) > 0:
			# here it await for a space separated columns
			pairs = line.strip().split()

			if sys.version_info[0] < 3:
				word = pairs[0].decode('utf-8')
			else:
				word = pairs[0]

			if number_normalized:
				word = normalize_word(word)
			#label = pairs[-1]
			words.append(word)
			#labels.append(label)
			word_Ids.append(word_alphabet.get_index(word))
			#print(words)
			#label_Ids.append(label_alphabet.get_index(label))
			## get features
			feat_list = []
			feat_Id = []
			for idx in range(feature_num):
				feat_idx = pairs[idx+1].split(']', 1)[-1]
				feat_list.append(feat_idx)
				feat_Id.append(feature_alphabets[idx].get_index(feat_idx))
			features.append(feat_list)
			features_Ids.append(feat_Id)
			
			## get char
			char_list = []
			char_Id = []
			for char in word:
				char_list.append(char)
			# WHAT DONT UNDERSTAND THE ROLE OF PADDING CHAR (not really important, by default, padding is set to -1 but thought)
			if char_padding_size > 0:
				char_number = len(char_list)
				if char_number < char_padding_size:
					char_list = char_list + [char_padding_symbol] * (char_padding_size - char_number)
				assert(len(char_list) == char_padding_size)
			else:
				### not padding
				pass
			for char in char_list:
				char_Id.append(char_alphabet.get_index(char))
			# append lst of chars and list of char_ids to respective super_list (over the whole input_file)
			chars.append(char_list)
			char_Ids.append(char_Id)
		## case blankline (end of sentence / instance), here it sets aside the sentences too long (len >= max_sent_length)
			#print(words, chars)
		else:
			if (len(words) > 0) and ((max_sent_length < 0) or (len(words) < max_sent_length)):
				# suppressed features and feature_Ids from these lists
				# suppressed labels because we are reading only deployment data (so a list of words)
				instance_texts.append([words, chars, features])
				instance_Ids.append([word_Ids, char_Ids, features_Ids])
			words = []
			features = []
			chars = []
			#labels = []
			word_Ids = []
			features_Ids = []
			char_Ids = []
			#label_Ids = []
	return instance_texts, instance_Ids


# !!!!!!TODO implement for the training
def build_pretrain_embedding(embedding_path, word_alphabet, embedd_dim = 100, norm = True):
	pretrain_emb = np.empty([word_alphabet.size(), embedd_dim])
	return pretrain_emb, embedd_dim
