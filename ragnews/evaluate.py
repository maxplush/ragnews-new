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
        db = ragnews.ArticleDB()
        textprompt = 'hellow world'
        output = ragnews.rag(textprompt, db)
        return output