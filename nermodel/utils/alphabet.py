# Mappig of objects (words, sequences, labels to integer)

import json
import os
import sys

class Alphabet:
	def __init__(self, name, label = False, keep_growing = True):
		self.name = name
		self.UNKNOWN = "</unk>"
		self.label = label
		self.instance2index = {}
		self.instances = []
		self.keep_growing = keep_growing

		# special role to the 0 index
		self.defaut_index = 0
		self.next_index = 1
		if not self.label:
			self.add(self.UNKNOWN)

	def clear(self, keep_growing = True):
		self.instance2index = {}
		self.instances = []
		self.keep_growing = keep_growing

		self.defaut_index = 0
		self.next_index = 1

	def add(self, instance):
		if instance not in self.instance2index:
			self.instances.append(instance)
			self.instance2index[instance] = self.next_index
			self.next_index += 1

	def get_index(self, instance):
		try:
			return self.instance2index[instance]
		except KeyError:
			if self.keep_growing:
				index = self.next_index
				self.add(instance)
				return index
			else:
				return self.instance2index[self.UNKNOWN]

	def get_instance(self, index):
		if index ==0:
			if self.label:
				return self.instances[0]
			else:
				return None
		try:
			return self.instances[index - 1]
		except IndexError:
			print('WARNING: Alphabet get_instance, unknwown instance, so return the first label')
			return self.instances[0]

	def size(self):
		# is it -1 instead of ncrfpp code + 1 
		return len(self.instances) + 1

	def iteritems(self):
		if sys.version_info[0] < 3:# if python < 3 , diferent access to items
			return self.instance2index.iteritems()
		else:
			return self.instance2index.items()

	def enumerate_items(self, start = 1):
		if start < 1 or start >= self.size():
			raise IndexError("Enumerate is allowed betwee [1 : alphabet_size]")
		else:
			return zip(range(start, len(self.instances) + 1), self.instances[start - 1:])

	def close(self):
		self.keep_growing = False

	def open(self):
		self.keep_growing = True

	def get_content(self):
		return {'instance2index': self.instance2index, 'instances': self.instances}

	def from_json(self, data):
		self.instances = data['instances']
		self.instance2index = data['instance2index']
		self.next_index = len(self.instances) + 1

	def save(self, output_directory, name = None):
		"save both alphabet records to a given directory"
		saving_name = name if name else self.__name
		try:
			json.dump(self.get_content(), open(os.path.join(output_directory, saving_name + ".json"), 'w'))
		except Exception as e:
			print("Exception: Alphabet s not saved " % repr(e))

	def load(self, input_directory, name = None):
		""" allow to use old models
		"""
		loading_name = name if name else self.__name
		self.from_json(json.load(open(os.path.join(input_directory, loading_name + ".json"))))