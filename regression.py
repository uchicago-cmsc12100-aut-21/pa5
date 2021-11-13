'''
Linear regression

YOUR NAME HERE

Main file for linear regression and model selection.
'''

import numpy as np
from sklearn.model_selection import train_test_split
import util


class DataSet(object):
    '''
    Class for representing a data set.
    '''

    def __init__(self, dir_path):
        '''
        Class for representing a dataset, performs train/test
        splitting.

        Inputs:
            dir_path: (string) path to the directory that contains the
              file
        '''

        parameters_dict = util.load_json_file(dir_path, "parameters.json")
        self.feature_idx = parameters_dict["feature_idx"]
        self.name = parameters_dict["name"]
        self.target_idx = parameters_dict["target_idx"]
        self.training_fraction = parameters_dict["training_fraction"]
        self.seed = parameters_dict["seed"]
        self.labels, data = util.load_numpy_array(dir_path, "data.csv")

        # do standardization before train_test_split
        if(parameters_dict["standardization"] == "yes"):
            data = self.standardize_features(data)

        self.training_data, self.testing_data = train_test_split(data,
            train_size=self.training_fraction, test_size=None,
            random_state=self.seed)

    # data standardization
    def standardize_features(self, data): 
        '''
        Standardize features to have mean 0.0 and standard deviation 1.0.
        Inputs:
          data (2D NumPy array of float/int): data to be standardized
        Returns (2D NumPy array of float/int): standardized data
        '''
        mu = data.mean(axis=0) 
        sigma = data.std(axis=0) 
        return (data - mu) / sigma

class Model(object):
    '''
    Class for representing a model.
    '''

    def __init__(self, dataset, feature_idx):
        '''
        Construct a data structure to hold the model.
        Inputs:
            dataset: an dataset instance
            feature_idx: a list of the feature indices for the columns (of the
              original data array) used in the model.
        '''

        # REPLACE pass WITH YOUR CODE
        pass

    def __repr__(self):
        '''
        Format model as a string.
        '''

        # Replace this return statement with one that returns a more
        # helpful string representation
        return "!!! You haven't implemented the Model __repr__ method yet !!!"

    ### Additional methods here


def compute_single_var_models(dataset):
    '''
    Computes all the single-variable models for a dataset

    Inputs:
        dataset: (DataSet object) a dataset

    Returns:
        List of Model objects, each representing a single-variable model
    '''

    # Replace [] with the list of models
    return []


def compute_all_vars_model(dataset):
    '''
    Computes a model that uses all the feature variables in the dataset

    Inputs:
        dataset: (DataSet object) a dataset

    Returns:
        A Model object that uses all the feature variables
    '''

    # Replace None with a model object
    return None


def compute_best_pair(dataset):
    '''
    Find the bivariate model with the best R2 value

    Inputs:
        dataset: (DataSet object) a dataset

    Returns:
        A Model object for the best bivariate model
    '''

    # Replace None with a model object
    return None


def forward_selection(dataset):
    '''
    Given a dataset with P feature variables, uses forward selection to
    select models for every value of K between 1 and P.

    Inputs:
        dataset: (DataSet object) a dataset

    Returns:
        A list (of length P) of Model objects. The first element is the
        model where K=1, the second element is the model where K=2, and so on.
    '''
    return []


def validate_model(dataset, model):
    '''
    Given a dataset and a model trained on the training data,
    compute the R2 of applying that model to the testing data.

    Inputs:
        dataset: (DataSet object) a dataset
        model: (Model object) A model that must have been trained
           on the dataset's training data.

    Returns:
        (float) An R2 value
    '''

    # Replace 0.0 with the correct R2 value
    return 0.0
