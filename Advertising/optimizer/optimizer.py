import math
import pandas as pd
pd.set_option('precision', 3)

import numpy as np




def optimizer(table):
    """
    Given a sub-campaigns matrix, resolve the problem optimization.
    :param table: sub-campaigns matrix, each cell contains the revenue

        EXAMPLE
        table = [
            [m_inf, 90, 100, 105, 110, m_inf, m_inf, m_inf],
            [0, 82, 90, 92, m_inf, m_inf, m_inf, m_inf],
            [0, 80, 83, 85, 86, m_inf, m_inf, m_inf],
            [m_inf, 90, 110, 115, 118, 120, m_inf, m_inf],
            [m_inf, 111, 130, 138, 142, 148, 155, m_inf]
        ]

    :return: list of bid-price-indexes correspondent to each sub-campaign
    """

    rows = len(table)

    # pointer matrix
    opt_indexes = []


    # optimization algorithm
    for row in range(0, rows):
        # copy the value of the first sub-campaign in the optimization table
        opt_table = table[row]

        max_value = np.max(opt_table)
        index = np.where(opt_table == max_value)

        opt_indexes.append(index)
        
    
    
    return opt_indexes

def get_optimizer_values(table, opt_indexes):
    values = []
    for row in range(len(opt_indexes)):
        values.append(table[row][opt_indexes[row]])
    return values