import time
import random 
from itertools import combinations
    
def pred_tree_d(clf, coalition, row, row_d):
    """
        The following function computes the characteristic function 
        We used the masking technique described in our report
        clf : ml model we used for predictions
        coalition : the features that constitute the coalition
        row : the observation for which we'll compute the banzhaf values
        row_d : our mask as explained in the report
    """
    hs = []
    for j in list(row.index):
        if j in coalition:
            hs.append(row[j])
        else:
            hs.append(row_d[j])
    return clf.predict([hs])

def make_value_function(clf, row, col, df):
    """
        Returns a function that computes the marginal gain of a feature
        clf : ml model we used for predictions
        row : observation
        col : column, that we want to determine its importanc
        df : the dataframe
    """
    def value(c):
        d = 1000 #pick a number 100 - 1000
        marginal_gain = 0
        for i in range(d):
            random.seed(int(time.time()) + i)
            ind = random.randint(0, df.shape[0] - 1)
            row_d = df[ind:ind+1].T.squeeze()
            marginal_gain += pred_tree_d(clf, c + [col], row, row_d) - pred_tree_d(clf, c, row, row_d)
        return marginal_gain/d
    return value

def make_coalitions(row, col):
    """
        The following function generates all possible coalitions of features exluding "col"
        row : the observation for which we cant to compute banzhaf values
        col : the col we exclude in the coalitions
    """
    rest = [x for x in row.index if x != col]
    for i in range(len(rest) + 1):
        for x in combinations(rest, i):
            yield list(x)
        
def compute_banzhaf(clf, row, col, df):
    """
        The following function computes the banzhaf values for a given observation "row", using the formula explicited in
        our report
        clf, row, col, df as described earlier
    """
    v = make_value_function(clf, row, col, df)
    result = sum([v(coal) / (2 ** (len(row) - 1)) for coal in make_coalitions(row, col)])
    return result
