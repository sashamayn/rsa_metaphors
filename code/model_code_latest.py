import numpy as np
from scipy.stats import norm
import csv, math
import sys
 
def priors_setup():
	#animals as keys
	priors_anim = {}
	#adjectives as keys
	priors_adj = {}
	animals = []
	adjectives = []

	with open('../data/priors.csv') as f:
		csv_reader = csv.reader(f, delimiter=',')
		line_count = 0
		for row in csv_reader:

			if line_count == 0:
				line_count+=1
				continue
			else:
				line_count+=1
				anim = row[1].strip()
				adj = row[2].strip()
				if anim == 'tiger' or adj in ['funny','happy','patient','reliable']:
					continue
				else:
					if anim in priors_anim:
						priors_anim[anim][adj] = (float(row[3])/100,float(row[4])/100)
					else:
						priors_anim[anim] = {adj:(float(row[3])/100,float(row[4])/100)}
						animals.append(anim)

					if adj in priors_adj:
						priors_adj[adj][anim] = (float(row[3])/100,float(row[4])/100)
					else:
						priors_adj[adj] = {anim:(float(row[3])/100,float(row[4])/100)}
						adjectives.append(adj)

	# print(animals,adjectives)
	return priors_anim, priors_adj, animals, adjectives

#https://towardsdatascience.com/kl-divergence-python-example-b87069e4b810
def kl_divergence(p, q):
	return np.sum(np.where(p != 0, p * np.log(p / q), 0))

class System:
	def __init__(self, l0_sd, l0_mean, p_h, p_match):
		self.priors_anim, self.priors_adj, self.animals, self.adjectives = priors_setup()
		self.l0_sd = float(l0_sd)
		# self.lamb = 1
		self.l0_mean = float(l0_mean)
		self.p_h = float(p_h)
		self.p_match = float(p_match)
		self.p_given_c = self.p_c()


	def p_c(self):
		p_given_c = {} #key: animal, value  - list (np-array?) of typicality values in the order in which they are in self.adjectives
		for each in self.animals:
			vec = []
			for adj in self.priors_anim[each]:
				vec.append(self.priors_anim[each][adj][0])
			p_given_c[each] = vec

		return p_given_c

	#pragmatic speaker
	#US1​​(PS1​​,PL0​​)=−KL(PS1​​∣∣PL0​​)
	#L_0 is uninformed

	#animal, adj
	#that's utility, not speaker
	def S1(self, anim, adj):
		s1Mean, s1SD = self.priors_anim[anim][adj]

		#how many do we need? 100?
		x = np.arange(0, 1, 0.01)
		pS1 = norm.pdf(x,loc=s1Mean,scale=s1SD)

		# pL0 = norm.pdf(x,loc=0.5,scale=self.l0_sd)
		pL0 = norm.pdf(x, loc=self.l0_mean, scale=self.l0_sd)

		return -kl_divergence(pS1,pL0)

	#pragmatic speaker
	'''
	intention = goal in Kao et al. = conversational context: specific (to communicate this feature) or general (uniform distribution)
	f - inferred feature
	cat - animal
	'''
	def L1(self, cat, utterance, intention, divergence_only=False):
		match = 1/len(self.adjectives)
		#category priors
		#since the only two we want to be considering for 
		#John is a shark are "shark" and "human". The rest are 0.
		if cat == 'human':
			p_c = self.p_h

		elif cat == utterance: 
			p_c = 1 - self.p_h

		else:
			return 0

		#priors over goals
		if intention == "na":
			goal_priors = [1/len(self.adjectives) for i in self.adjectives]
		else:
			goal_priors = [self.p_match if i==intention else (1-self.p_match)/(len(self.adjectives)-1) for i in self.adjectives]
		
		if divergence_only:
			l1 = [p_c * self.S1(utterance,self.adjectives[i]) for i in range(len(self.adjectives))]
		else:
			l1 = [p_c * goal_priors[i] * self.p_given_c[cat][i] * self.S1(utterance,self.adjectives[i]) for i in range(len(self.adjectives))]

		return l1

		#prepares the numbers and computes the predictions of L1
	def meta_l(self, utterance, intention):

		L1_matrix = np.empty(shape=(len(self.adjectives),2))

		cats = [utterance,"human"]
		for column in cats:
			L1_matrix[:,cats.index(column)] = self.L1(column,utterance,intention)

		L1_matrix /= L1_matrix.sum()

		return L1_matrix
	'''
	given a human prediction, return its probability
	utterance, conversational context (e.g. (bear_strong)) 
	get probability of what
	humans said
	'''
	# humanPrediction,utterance,context
	#also write of all predictions, rank: is it the 1st/2nd/3rd best etc.
	def compareToHuman(self):
		filename = '_'.join(['15.04_',str(self.l0_sd),str(self.l0_mean),str(self.p_h),str(self.p_match),'.csv'])
		model_predictions = open('predictions/'+filename,'w',newline='')
		writer = csv.writer(model_predictions)
		writer.writerow(["ChosenFeature","Adj","Animal","ModelProb","ModelRank"])
		human_predictions = open('../data/predictions.csv','r')

		reader = csv.reader(human_predictions, delimiter=",")
		for row in reader:
			if row[0]=="ChosenFeature":
				continue
			else:
				#the uttered animal
				utterance = row[-1]
				
				#the question - either the adj or 'na' for vague
				context = row[-2]

				#inferring that that's what the speaker means
				prediction = row[0]

				L1_matrix = self.meta_l(utterance, context)

				feat = self.adjectives.index(prediction)

				#1 for human
				model_prob = L1_matrix[feat,1]

				L1_probs = {}

				for r in range(L1_matrix.shape[0]):
					for c in range(L1_matrix.shape[1]):
						L1_probs[(r,c)]=L1_matrix[r,c]

				ranks = {key: rank for rank, key in enumerate(sorted(L1_probs, key=L1_probs.get, reverse=True), 1)}

				writer.writerow([prediction,context,utterance,model_prob,ranks[(feat,1)]])


def main():
	l0_sd, l0_mean, p_h, p_match = sys.argv[1:]
	s = System(l0_sd, l0_mean, p_h, p_match)

	s.compareToHuman()
	
main()