'''
Alexandra Mayn's final project 
for Recent Advances in Discourse Processing
'''

import numpy as np
import csv, math
 
#features - rows, animals - colums. normalize it
def typicalities_setup():
	#pretend that these are the only animals
	animals = []
	features = []
	typicality_matrix = np.empty(shape=[10,7])

	with open('typicalities.csv') as f:
		csv_reader = csv.reader(f, delimiter=',')
		line_count = 0
		for row in csv_reader:
			if line_count == 0:
				animals.extend(row[1:])
			else:
				features.append(row[0])
				for typ in range(1,len(row)):
					#normalized typicality
					typicality_matrix[line_count-1,typ-1]=int(row[typ])/6
			# features.append(row[0])
			line_count+=1
	#print(typicality_matrix)
	return typicality_matrix, animals, features

class System:
	def __init__(self):
		self.typicality_matrix, self.animals, self.features = typicalities_setup()
		self.salience_matrix = abs(self.typicality_matrix - 0.5) ** 4
		self.salience_matrix[np.where(self.salience_matrix == 0)] = 0.01
		self.salience_matrix /= self.salience_matrix.sum(axis=1, keepdims=True)
		#print(self.salience_matrix)

		#literal listener
	def L0(self, cat, f, utterance):
		if cat != utterance:
			return 0
		else:
			feat = self.features.index(f)
			anim = self.animals.index(utterance)
			return self.salience_matrix[feat,anim]

	#pragmatic speaker
	def S1(self, f, utterance, lamb):
		l0 = self.L0(utterance,f,utterance)
		return l0**lamb

	#pragmatic speaker
	'''
	intention = goal in Kao et al.; 
	specific (to communicate this feature; p = 0.5)
	or general (uniform distribution)
	'''
	def L1(self, cat, f, p_fi, utterance, lamb):
		

		s1 = self.S1(f, utterance, lamb)

		#category priors
		if cat == 'human':
			p_c = 0.99
		else: 
			p_c = 0.01

		p_fc = self.salience_matrix[self.features.index(f),self.animals.index(cat)]
		return p_c*p_fc*p_fi*s1

		#prepares the numbers and computes the predictions of L1
	def meta_l(self, utterance, intention, lamb):
		if intention in self.features:
			intentions = [
				0.5 if f == intention else 0.5 / (len(intention) - 1)
				for f in self.features
			]
		else:
			intentions = [1 / len(intention) for f in self.features]


		L1_matrix = np.empty(shape=self.typicality_matrix.shape)
		for row in range(L1_matrix.shape[0]):
			for column in range(L1_matrix.shape[1]):
				L1_matrix[row,column] = self.L1(
					self.animals[column],
					self.features[row],
					intentions[row],
					utterance,
					lamb)

		prob = L1_matrix.max()
		r, c = np.where(L1_matrix == prob)
		
		for posmaxfeature in r:
			if self.features[posmaxfeature] == intention:
				r = posmaxfeature
				break
		else:
			r = r[0]
		if self.animals.index('human') in c:
			c = self.animals.index('human')
		else:
			c = c[0]
		feat = self.features[r]
		anim = self.animals[c]
		typicality = self.typicality_matrix[r, self.animals.index(utterance)]

		return anim, feat, prob, typicality


def main():
	s = System()
	lamb = 1

	for intention in s.features + ['uniform']:
		for utterance in s.animals:
			anim, feat, prob, typ = s.meta_l(utterance, intention, lamb)
			print("The person wanted to communicate the feature:",intention.upper())
			print("To do that, she uttered:",utterance.upper())
			print("The model's inference was that she meant", anim.upper(), "with the feature ", feat.upper())
			print("Probability of the prediction: ", round(prob,4))
			print("Typicality of the feature ", feat.upper(), " for ", utterance.upper(), ": ", round(typ,4),"\n")

main()