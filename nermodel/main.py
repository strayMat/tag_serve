from utils.data import Data
from ner_model import train, data_initialization, load_model_decode
import torch

def myTrain(confdict):
	print('Model Train')
	data = Data()
	data.read_config(confdict)
	data.HP_gpu = torch.cuda.is_available()
	data_initialization(data)
	data.generate_instance('train')
	data.generate_instance('dev')
	data.generate_instance('test')
	data.build_pretrain_emb()
	f1 = train(data)
	return f1

# decoding with a test file containing labels (test purpose usually)
def myDecode(confdict, verbose=True):
	data = Data()
	data.read_config(confdict)
	print('Model Decode')
	try:
		data.load(data.dset_dir)
	except TypeError:
		data.load_export(data.xpt_dir)
	data.read_config(confdict)
	data.HP_gpu = torch.cuda.is_available()
	print('Decoding source: ', data.raw_dir)
	if verbose:
		data.show_data_summary()
	data.generate_instance('raw')
	decode_results, pred_scores = load_model_decode(data, 'raw')
	if data.decode_dir:
		data.write_decoded_results(decode_results, 'raw')

