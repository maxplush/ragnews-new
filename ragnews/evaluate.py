'''
This file is for evaluating the quality of our RAG system 
using the Hairy Trumpet tool/dataset.
'''
import sys
sys.path.append('/Users/maxplush/Documents/ragnews-new')

import ragnews

class RAGEvaluator:
    def __init__(self, valid_labels):
        '''
        Initialize the RAGClassifier. The __init__ function should take
        an input that specifies the valid labels to predict.
        The class should be a "predictor" following the scikit-learn interface.
        You will have to use the ragnews.rag function internally to perform the prediction.
        '''
        self.valid_labels = valid_labels

    def predict(self, masked_text):
        '''
        >>> model = RAGEvaluator()
        >>> model.predict('There no mask token here.')
        []
        >>> model.predict('[MASK0] is the democratic nominee')
        ['Harris']
        >>> model.predict('[MASK0] is the democratic nominee and [MASK1] is the republican nominee')
        ['Harris', 'Trump']
        '''
        # you might think about:
        # calling the ragnews.run_llm function directly;
        # so we will call the ragnews.rag function

        valid_labels = ['Harris', 'Trump']

        db = ragnews.ArticleDB('ragnews.db')
        textprompt = f'''
This is a fancier question that is based on standard cloze style benchmarks.
I'm going to provide you a sentence, and that sentence will have a masked token inside of it: [MASK0].
And your job is to tell me what the value of that masked token was.
Valid values include: {valid_labels}

If the answer is a name return only the last name. 
Return only one word!
You should not provide any explanation or other extraneous words.

INPUT: [MASK0] is the democratic nominee
OUTPUT: Harris

INPUT: [MASK0] is the democratic nominee and [MASK1] is the republican nominee
OUTPUT: Harris Trump

INPUT: {masked_text}
OUTPUT: '''
        output = ragnews.rag(textprompt, db, keywords_text=masked_text)
        return output
    

    # tweak hyperparmaters 
    # goal is to run through the data points on 70% acccuracy 
    # implment a for loop that calls a predict function on each line of that file, checks to see if is correct
    # to see if our version is better for the predict function

    # also maybe add this

    # import logging
    # logging.basicConfig(
    # format='%(asctime)s %(levelname)-8s %(message)s',
    # datefmt='%Y-%m-%d %H:%M:%S',
    # level=logging.INFO,
    # )

    # add the logging
    # steps to run 
    # ensure you are on evaluate branch and ragnews-new directory
    # python3 -i ragnews/evaluate.py                     
    # model = RAGEvaluator()
    # model.predict('[MASK0] is the democratic nominee')



if __name__ == '__main__':
    import argparse
    import json
    import os

    filepath = r"hairy-trumpet/data/wiki__page=2024_United_States_presidential_election,recursive_depth=0__dpsize=paragraph,transformations=[canonicalize, group, rmtitles, split]"

    # Argument parser for command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default=filepath)
    args = parser.parse_args()

    # Load data from the current file
    with open(args.data, 'r') as f:
        data = [json.loads(line) for line in f]

    # Extract unique labels from the data
    labels = set()
    with open(args.data, 'r') as fin:
        for i, line in enumerate(fin):
            dp = json.loads(line)
            labels.update(dp['masks'])

    n_correct = 0
    n_tests = 10
    print(labels)
    evaluator = RAGEvaluator(valid_labels=labels)
    for i, text_case in enumerate(data[:n_tests]):
        prediction = evaluator.predict(text_case["masked_text"])
        if prediction == text_case["masks"][0]:
            n_correct += 1
        print(prediction)
        print(text_case["masks"])

    print("Number Correct:", n_correct)
    print("Total Test Cases:", n_tests)


    # print(f"Processed {args.data}: {(data[0])} data points and {(labels)} unique labels.")


    # TODO: Need to improve accuracy and also update readme to reflect running code 
    # python3 ragnews/evaluate.py --data "/Users/maxplush/Documents/ragnews-new/hairy-trumpet/data/wiki__page=2024_United_States_elections,recursive_depth=1__dpsize=sentence,transformations=[canonicalize, group, rmtitles, split]"
    # could improve system prompts or other hyper paremeters




    
