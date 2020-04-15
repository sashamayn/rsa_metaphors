This is a work in progress extending Kao et al.'s metaphor model to continuous semantics.

The .pdf document 'metaphor_paper.pdf' is the paper itself in which the model is proposed. It outlines behavioral experiments that were conducted to evaluate the model's predictions. There's also a poster presented at ESSLLI'19 summarizing the work. Note: the model has changed a lot since this draft was written; an updated version of the paper to come soon. 

The .py executable 'metaphor_source.py' is the implementation of the proposed model. It takes in the priors (referred to as the 'demo dataset' in the paper) contained in 'typicalities.csv'. It does not take any command line arguments. To run it, type in the terminal window: python3 metaphor_source.py. It will print out the intepretations for each intention-utterance pair. For example output, see Appendix 7.2 in the paper.

The 'code' folder contains the model itself as well as an R script which analyzes the model's predictions and compares them to the experimentally collected priors (contained in the 'data' folder).

