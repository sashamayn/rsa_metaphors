The .pdf document 'metaphor_paper.pdf' is the paper itself in which the model is proposed. It is a work in progress and outlines behavioral experiments that are being run to evaluate the model's predictions.

The .py executable 'metaphor_source.py' is the implementation of the proposed model. It takes in the priors (referred to as the 'demo dataset' in the paper) contained in 'typicalities.csv'. It does not take any command line arguments. To run it, type in the terminal window: python3 metaphor_source.py. It will print out the intepretations for each intention-utterance pair. For example output, see Appendix 7.2 in the paper.

Here are a couple of examples of the model's predictions. 

In the example below, 'lazy' has 0 typicality for ants (see the dataset), therefore it has high salience. The example below is therefore in the specific, low salience condition and corresponds to the experimental data point "Is John lazy? John is an ant." Probability displayed is the maximum probability of an interpretation for this intention-utterance pair. The model predicts that the speaker intends to say that John is a human who is not at all lazy.

Conversational context: LAZY
Speaker's utterance: ANT
Model's inference is HUMAN with the feature  LAZY
Probability of the prediction:  0.0251
Typicality of the feature  LAZY  for  ANT :  0.0
Salience of the feature  LAZY  for  ANT :  0.2093

The example below is from the vague condition, so it is not clear to the listener what feature the speaker wanted to communicate. This corresponds to the experimental datapoint "What is John like? He is a dog." The model predicts the interpretation that John is happy since happy is the most salient (and typical) feature for a dog.

Conversational context: UNIFORM
Speaker's utterance: DOG
Model's inference is HUMAN with the feature  HAPPY
Probability of the prediction:  0.0055
Typicality of the feature  HAPPY  for  DOG :  1.0
Salience of the feature  HAPPY  for  DOG :  0.225
