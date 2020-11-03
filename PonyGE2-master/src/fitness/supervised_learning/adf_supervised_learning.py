import numpy as np
np.seterr(all="raise")
from os import path
from algorithm.parameters import params
from utilities.fitness.get_data import get_data
from utilities.fitness.math_functions import *
from utilities.fitness.optimize_constants import optimize_constants

from fitness.base_ff_classes.base_ff import base_ff


class adf_supervised_learning(base_ff):
    """
    Fitness function for supervised learning, ie regression and
    classification problems. Given a set of training or test data,
    returns the error between y (true labels) and yhat (estimated
    labels).

    We can pass in the error metric and the dataset via the params
    dictionary. Of error metrics, eg RMSE is suitable for regression,
    while F1-score, hinge-loss and others are suitable for
    classification.

    This is an abstract class which exists just to be subclassed:
    should not be instantiated.
    """

    def __init__(self):
        # Initialise base fitness function class.
        super().__init__()

        # # Get training and test data
        # self.training_in, self.training_exp, self.test_in, self.test_exp = \
        #     get_data(params['DATASET_TRAIN'], params['DATASET_TEST'])
        #
        # # Find number of variables.
        # self.n_vars = np.shape(self.training_in)[0]
        #
        # # Regression/classification-style problems use training and test data.
        # if params['DATASET_TEST']:
        #     self.training_test = True

    @staticmethod
    def multiple_operation(v, n):
        result = np.sum(v)
        return np.greater_equal(result, n)

    # def multiple_operation(f, v, n):
    #     result = f[0]
    #     for i in range(len(v)):
    #         if v[i]:
    #             result = np.add(result, f[i + 1])
    #         else:
    #             result = np.subtract(result, f[i + 1])
    #
    #     return np.greater_equal(result, n)

    def evaluate(self, ind, **kwargs):
        """
        Note that math functions used in the solutions are imported from either
        utilities.fitness.math_functions or called from numpy.

        :param ind: An individual to be evaluated.
        :param kwargs: An optional parameter for problems with training/test
        data. Specifies the distribution (i.e. training or test) upon which
        evaluation is to be performed.
        :return: The fitness of the evaluated individual.
        """
        x = kwargs['x']
        y = kwargs['y']

        results = []

        p, d = ind.phenotype, {'np': np, 'multiple_operation': self.multiple_operation}

        for i in range(x.shape[0]):
            d['x'] = x[i, :]
            exec(p, d)
            y_prev = d['result']
            assert np.isrealobj(y_prev)
            results.append(y[i] - y_prev)

        n_match = results.count(0)
        return n_match/len(results)


