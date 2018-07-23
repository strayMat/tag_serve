from utils.data import Data
from ner_model import train, data_initialization
import torch

def mytrain(confdict):
	data = Data()
	data.read_config(confdict)
	data.HP_gpu = torch.cuda.is_available()
	data_initialization(data)
	data.generate_instance('train')
	data.generate_instance('dev')
	data.generate_instance('test')
	data.build_pretrain_emb()
	train(data)

def myDecode(confdict):
	data = Data()
