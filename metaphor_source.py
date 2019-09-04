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
		mu = 0.5
		kappa = 4
		self.typicality_matrix, self.animals, self.features = typicalities_setup()
		self.salience_matrix = abs(self.typicality_matrix - mu) ** kappa
		self.salience_matrix[np.where(self.salience_matrix == 0)] = 0.01
		self.salience_matrix /= self.salience_matrix.sum(axis=0, keepdims=True) 
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
	intention = goal in Kao et al. = conversational context:
	specific (to communicate this feature)
	or general (uniform distribution)
	'''
	def L1(self, cat, f, utterance, lamb, joint_matrix, intention):

		s1 = self.S1(f, utterance, lamb)

		#category priors
		if cat == 'human':
			p_c = 0.99
		else: 
			p_c = 0.01

		row = self.features.index(f)

		if intention == 'uniform': #last column
			col = np.size(joint_matrix,1)-1
		else:
			col = self.features.index(intention)

		p_fi_given_c = joint_matrix[row,col]

		return p_c*s1*p_fi_given_c

		#prepares the numbers and computes the predictions of L1
	def meta_l(self, utterance, intention, lamb):
		'''
		n_match - parameter in p(f,i|c) when f=i
		n_mismatch - parameter in p(f,i|c) when f!=i
		'''
		n_match = 4
		n_mismatch = 0.25


		L1_matrix = np.empty(shape=self.typicality_matrix.shape)
		num_cols = np.size(self.typicality_matrix,0)
		for column in range(L1_matrix.shape[1]):
			joint_matrix = np.array([self.salience_matrix[:,column],]*(num_cols+1)).transpose()

			for rj in range(joint_matrix.shape[0]):
				for cj in range(joint_matrix.shape[1]):
					if rj == cj:
						joint_matrix[rj,cj]*=n_match
					elif cj!=(joint_matrix.shape[1]-1):
						joint_matrix[rj,cj]*=n_mismatch

						#renormalize
			joint_matrix /= np.sum(joint_matrix)

			for row in range(L1_matrix.shape[0]): #for each feature; can switch?
				L1_matrix[row,column] = self.L1(
					self.animals[column],
					self.features[row],
					utterance,
					lamb,
					joint_matrix, intention)


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
		typ = self.typicality_matrix[r, self.animals.index(utterance)]
		sal = self.salience_matrix[r,self.animals.index(utterance)]

		return anim, feat, prob, typ, sal, self.salience_matrix


def main():
	s = System()
	lamb = 1

	for intention in s.features + ['uniform']:
		for utterance in s.animals:
			anim, feat, prob, typ, sal, salience_matrix = s.meta_l(utterance, intention, lamb)
			
			print("Conversational context:",intention.upper())
			print("Speaker's utterance:",utterance.upper())
			print("Model's inference is", anim.upper(), "with the feature ", feat.upper())
			print("Probability of the prediction: ", round(prob,4))
			print("Typicality of the feature ", feat.upper(), " for ", utterance.upper(), ": ", round(typ,4))
			print("Salience of the feature ", feat.upper(), " for ", utterance.upper(), ": ", round(sal,4),"\n")
	
main()