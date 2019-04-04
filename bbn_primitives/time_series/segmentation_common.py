import abc
import numpy as np
import scipy

from d3m.container import ndarray as d3m_ndarray
from d3m.container import List
from numpy import ndarray

#__all__ = ('CurveFitting', 'CurveFittingWithData', 'applyFitting')


#class CurveFitting(abc.ABC):
#    """
#    This class encapsulates output of curve fitting applied for segment representation
#    """
#
#    def __init__(self, deg = 0, beta = None, sigma = None, N = None):
#        self.beta = beta
#        self.sigma = sigma
#        self.N = N
#
#
#class CurveFittingWithData(abc.ABC):
#    """
#    This class encapsulates an instance of CurveFitting along with corresponding data
#    """
#
#    def __init__(self, curve_fitting = None, data = None):
#        self.curve_fitting = curve_fitting
#        self.data = data

def applyFitting(n: int, fitting_params: ndarray) -> ndarray:
    # TODO: return value depends only on n (int), there's typically limited number
    # of possible values and output might be cached
    deg, num_feats = fitting_params.shape

    x = np.linspace(0., 1., n)
    y = np.zeros((n, num_feats))
    for d in range(num_feats):
        pfcn = np.poly1d(fitting_params[:, d])
        y[:, d] = pfcn(x)

    return y

#def seq_vocab(seq: List[List[d3m_ndarray]]) -> dict:
def seq_vocab(seq: List) -> dict:
    all_set = list(set([ tuple(x) for y in seq for x in y ]))
    vocab = { all_set[i]:i for i in range(len(all_set)) }
    return vocab

#def seq_to_tokenfreq_csr(seq: List[List[d3m_ndarray]], vocab: dict = None) -> d3m_ndarray:
def seq_to_tokenfreq_csr(seq: List, vocab: dict = None) -> ndarray:
    X = list()
    Y = list()
    data = list()
    for i in range(len(seq)):
        if seq[i].size == 0:
            continue

        a = np.unique(seq[i], axis=0, return_counts=True)
        if vocab is not None:
            a = np.array([
                  [ vocab[tuple(a[0][j])], a[1][j] ] for j in range(len(a[0])) if tuple(a[0][j]) in vocab
                ])
            if len(a) == 0:
                continue

        Y.append(a[:,0])
        X.append(np.ones_like(a[:,0], dtype=int)*i)
        data.append(a[:,1])

    X = np.hstack(X)
    Y = np.hstack(Y)
    data = np.hstack(data)

    return scipy.sparse.coo_matrix((data, (X,Y)), shape=(len(seq), len(vocab)))


