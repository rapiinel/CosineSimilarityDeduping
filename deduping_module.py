
# importing dependencies
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sparse_dot_topn import awesome_cossim_topn 
import re

from sklearn.feature_extraction.text import TfidfVectorizer



class deduping_class():

    def __init__(self, gt, nm):
        self.ground_truth = gt
        self.to_match = nm


    def 
