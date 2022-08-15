import numpy as np
import numpy.matlib as npm
from . import Normalization

"""
Special function getting bases for probability simplex (sum to one)
"""
def ProbSimplex(nS):
    A = npm.ones(nS)
    b = np.array([[1]])

    base,feas = Normalization.GaussElim(A,b)
    q = Normalization.GramSchmidt(base)

    return q
