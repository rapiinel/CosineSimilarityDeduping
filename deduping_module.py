
# importing dependencies
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sparse_dot_topn import awesome_cossim_topn 
import re

from sklearn.feature_extraction.text import TfidfVectorizer



class deduping_class():

    def __init__(self, gt):
        """
        Input: reference/groundtruth
        
        """
        self.ground_truth = gt
        self.ngrams_value = None

    def key_selector(self, nm, *column_names):
        """
        This function lets the user select the needed columns to be keys.
        To make this a simple function, the column names of the ground truth input and dataframe to match should be equal

        input: 
            dataframe to be matached
            List of columns to be used as keys
        output: 
            sets the key reference column in ground truth
            return 
        """

        self.ground_truth['primary_key'] = self.ground_truth[column_names].values.tolist()
        self.ground_truth['primary_key'] = self.ground_truth['primary_key'].apply(''.join)

        nm['primary_key'] = nm[column_names].values.tolist()
        nm['primary_key'] = nm['primary_key'].apply(''.join)

        return nm

    def ngrams(self, string, n=3):
        """
        This is a supporting function for the vectorizer
        input:
            n, n is the number of ngrams
        """
        if self.ngrams_value != None:
            n = self.ngrams_value

        string = re.sub(r'[,-./]|\sBD',r'', string)
        ngrams = zip(*[string[i:] for i in range(n)])
        return [''.join(ngram) for ngram in ngrams]

    def set_ngrams(self, n):
        """
        If user want to change the number of n in ngrams function
        """
        self.ngrams_value = n

    def vectorizer(self, nm):
        """
        This function will convert the dataframe into a sparse matrix
        input: 
            nm, the dataframe that needs to be matched
        output:
            the result will be stored as an attribute
            self.nm_tfidf
            self.gt_tfidf
        """
        vectorizer = TfidfVectorizer(min_df=1, analyzer=self.ngrams)
        combined_list = nm['primary_key'].tolist() + self.gt['primary_key'].tolist()
        vectorizer.fit(nm['primary_key'].tolist() + self.gt['primary_key'].tolist())

        self.nm_tfidf = nm['primary_key'].tolist()
        self.nm_tfidf = vectorizer.transform(self.nm_tfidf)

        self.gt_tfidf = self.gt['primary_key']
        self.gt_tfidf = vectorizer.transform(self.gt_tfidf)