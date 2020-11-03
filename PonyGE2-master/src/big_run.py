#! /usr/bin/env python

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from stats.stats import get_stats
from algorithm.parameters import params, set_params
from algorithm.search_loop import new_search_loop
import sys


def mane():
    individuals = new_search_loop()
    get_stats(individuals, end=True)


if __name__ == "__main__":
    set_params(sys.argv[1:])
    params['SEARCH_LOOP'] = 'new_search_loop'

    mane()

