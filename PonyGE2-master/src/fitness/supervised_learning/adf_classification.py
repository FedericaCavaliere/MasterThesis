from fitness.supervised_learning.adf_supervised_learning import adf_supervised_learning

from algorithm.parameters import params
from utilities.fitness.error_metric import f1_score


class adf_classification(adf_supervised_learning):
    """Fitness function for classification. We just slightly specialise the
    function for supervised_learning."""

    def __init__(self):
        # Initialise base fitness function class.
        super().__init__()

        # Set error metric if it's not set already.
        if params['ERROR_METRIC'] is None:
            params['ERROR_METRIC'] = f1_score

        self.maximise = params['ERROR_METRIC'].maximise
