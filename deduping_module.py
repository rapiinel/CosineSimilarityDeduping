
# importing dependencies
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sparse_dot_topn import awesome_cossim_topn 
import re
import json

from sklearn.feature_extraction.text import TfidfVectorizer



class deduping_class():

    def __init__(self, gt, object):
        """
        Input: reference/groundtruth, object to be deduped
        """
        self.ground_truth = gt
        # dropping salesforce footer in the extracted file
        self.ground_truth.dropna(subset=['Salesforce Contact Id'], inplace= True)
        self.ground_truth.reset_index(drop=True, inplace= True)
        self.ngrams_value = None
        self.state_reference_initiator()
        if str(object).lower() == 'account':
            self.ground_truth.drop_duplicates(subset='Salesforce Account Id', inplace= True)
            self.ground_truth.reset_index(drop=True, inplace= True)
        
    
    def state_reference_initiator(self, link = 'state_reference.json'):
        f = open(link)
        self.state_reference = json.load(f)
        f.close()


    def key_selector(self, *column_names, data = None):
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

        if str(type(data)) == "<class 'NoneType'>":
            raise ValueError("There is no dataframe inputed for matching")
        else:
            self.nm = data

        #column checker
        for col_value in column_names:
            if col_value not in self.ground_truth.columns:
                raise KeyError(f"{col_value}, Column name not found in ground truth dataframe")

        for col_value in column_names:
            if col_value not in self.nm.columns:
                raise KeyError(f"{col_value}, Column name not found in name to match dataframe")
        
        column_names = [value for value in column_names]

        # making sure the selected columns in string format
        for dataframe in [self.nm, self.ground_truth]:
            for col in column_names:
                dataframe[col] = dataframe[col].astype('str')
                dataframe[col] = dataframe[col].str.replace(" ", "")

        self.ground_truth['primary_key'] = self.ground_truth[column_names].values.tolist()
        self.ground_truth['primary_key'] = self.ground_truth['primary_key'].apply(''.join)

        self.nm['primary_key'] = self.nm[column_names].values.tolist()
        self.nm['primary_key'] = self.nm['primary_key'].apply(''.join)

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

    def vectorizer(self):
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
        self.combined_list = self.nm['primary_key'].tolist() + self.ground_truth['primary_key'].tolist()
        vectorizer.fit(self.nm['primary_key'].tolist() + self.ground_truth['primary_key'].tolist())

        self.nm_tfidf = self.nm['primary_key'].tolist()
        self.nm_tfidf = vectorizer.transform(self.nm_tfidf)
        self.nm_tfidf_df = pd.DataFrame.sparse.from_spmatrix(self.nm_tfidf, columns= vectorizer.get_feature_names_out())

        self.gt_tfidf = self.ground_truth['primary_key']
        self.gt_tfidf = vectorizer.transform(self.gt_tfidf)
        self.gt_tfidf_df = pd.DataFrame.sparse.from_spmatrix(self.gt_tfidf, columns= vectorizer.get_feature_names_out())

    def get_match(self, top):
        """
        This function will match the ground truth and to match dataframe tfidf format

        input: the input will only be self and the number matched record, just make sure that you already run the vecotrizer

        output: Will store the output to self.matches
        """

        matches = awesome_cossim_topn(self.nm_tfidf, self.gt_tfidf.transpose(), 10, 0.8, use_threads=True, n_jobs=6)

        self.matched = self.get_matches_df(matches, self.combined_list, top=top)

        self.segment_output()

    def get_matches_df(self,sparse_matrix, name_vector, top=100):
        """
        This is an internal function that converts the sparse matrix back into a dataframe format
        """
        non_zeros = sparse_matrix.nonzero()
        
        sparserows = non_zeros[0]
        sparsecols = non_zeros[1]
        
        if top & top < sparsecols.size:
            nr_matches = top
        else:
            print("The top value is not set or the value exceeds the nonzero size")
            nr_matches = sparsecols.size

        left_side = np.empty([nr_matches], dtype=object)
        index_value = np.empty([nr_matches], dtype=object)
        right_side = np.empty([nr_matches], dtype=object)
        contact_id = np.empty([nr_matches], dtype=object)
        similairity = np.zeros(nr_matches)
        
        for index in range(0, nr_matches):
            left_side[index] = self.nm.loc[sparserows[index], 'primary_key']
            index_value[index] = sparserows[index]
            right_side[index] = self.ground_truth.loc[sparsecols[index], 'primary_key']
            contact_id[index] = self.ground_truth.loc[sparsecols[index], 'Salesforce Contact Id']
            similairity[index] = sparse_matrix.data[index]
        
        return pd.DataFrame({'index':index_value,
                            'Matched DataFrame Key': left_side,
                            'Ground Truth Key': right_side,
                            'Ground Truth ID':contact_id, 
                            'similarity': similairity})

    def state_abbrev(self, value):
        """
        This function will transform the state into abbrevation format
        """
        value = str(value).lower() # making sure the value is in string format

        try:
            if value != 'nan':
                return self.state_reference[value]
            else:
                return value
        except:
            return "Error no value in reference"
    
    def segment_output(self):
        #non_matched
        self.non_matched_output = self.nm[~self.nm.index.isin(self.matched['index'].tolist())]
        #matched
        self.matched_output = self.nm.merge(self.matched[['index','Ground Truth ID','similarity']].set_index('index'), left_index= True, right_index=True)