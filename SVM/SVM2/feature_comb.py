import os
import sys
import itertools
import numpy as np

def combination_gen():

    feature_dict = {"1PD": 4, "1PF":5, "1QD": 6, "1QF": 7, "SPEED": 10, "PS_WORD": 13, "SIL_AVG": 14}
    comb_set = []
      
    combination = [[4, 10], [5, 10], [6, 10], [7, 10], [4,14],
                   [4, 5, 10], [4, 7, 10], [5, 6, 10], [6,7,10],[4,10,13],[4,5,14],[4,7,14],[4,10,14],[6,10,14],
                   [4, 10, 13, 14], [6, 10, 13, 14], [4,5,10,14], [4,7,10,14],
                   [4, 5, 10, 13, 14], [4, 7, 10, 13, 14], [5,6,10,13,14],[6,7,10,13,14],
                   [4,5,7,10,14], [4,6,10,14], [4,5,7,14], [4,5,6,10,13,14],[4,5,6,7,10],
                   [4,6,7,10,13,14], [4,5,7,10,13,14],[4,5,6,7,13,14],[4,5,6,7,10,14],[4,5,6,7,10,13],
                   [4,5,6,7,10,13,14],
                   [5,7,13,14], [5,7,14], [5,7,10,13,14],[5,7,10,13],[5,7,10,14],
                   [5,7,10],[5,10,13,14],[5,10,14],[7,10,13,14],
                   [7,10,14],[10,13,14],
                   [4,5],[4,6],[4,7],[4,13],[5,6],[5,7],[5,13],[5,14],[6,7],[6,13],[6,14],[7,13],[7,14],[10,13],[10,14],[13,14],
		   [4], [5], [6], [7], [10], [13], [14]
                   ]
    
    # combination = [[4], [5], [6], [7], [10], [13], [14]]

    """
    for i in range(7):

        for comb in itertools.combinations([4,6,7,10,13,14], i+1):

            comb_set.append(list(comb))

    """

    for i in combination:
        comb_set.append(i)

    return comb_set

