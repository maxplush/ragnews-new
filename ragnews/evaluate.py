'''
This file is for evaluating the quality of our RAG system 
using the Hairy Trumpet tool/dataset.
'''
import sys
sys.path.append('/Users/maxplush/Documents/ragnews-new')

import ragnews

class RAGEvaluator:
    def predict(self, x):
        '''
        >>> model = RAGEvaluator()
        >>> mdoel.predict('There no mask token here.')
        []
        '''
        db = ragnews.ArticleDB('ragnews.db')
        textprompt = 'hellow world'
        output = ragnews.rag(textprompt, db)
        return output
    

    #tweak hyperparmaters 
    # goal is to run through the data points on 70% acccuracy 
    # implment a for loop that calls a predict function on each line of that file, checks to see if is correct
    # to see if our version is better for the predict function