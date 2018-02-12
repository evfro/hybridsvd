import numpy as np
from scipy.sparse import csr_matrix

def evaluate_hits(test_data, key, target, recommendations):
    # this assumes that test_data dataframe and recommendations matrix
    # are aligned on and sorted by the "key"
    n_observations = test_data.shape[0]
    n_keys, topn = recommendations.shape
    rank_arr = np.arange(1, topn+1, dtype=np.min_scalar_type(topn))
    recs_rnk = np.lib.stride_tricks.as_strided(rank_arr, (n_keys, topn), (0, rank_arr.itemsize))

    dtype = np.bool
    shape = (n_keys, max(recommendations.max(), test_data[target].max())+1)
    eval_matrix = csr_matrix(shape, dtype=dtype)
    rank_matrix = csr_matrix(shape, dtype=rank_arr.dtype)

    # setting data and indices manually to avoid index dtype checks
    # and thus prevent possible unnecesssary copies of indices
    eval_matrix.data = np.ones(n_observations, dtype=dtype)
    eval_matrix.indices = test_data[target].values
    eval_matrix.indptr = np.r_[0, np.where(np.diff(test_data[key].values))[0]+1, n_observations]
    
    # support models that may generate < top-n recommendations
    # such models generate self._pad_const, which is negative by convention
    valid_recs = recommendations >= 0
    if not valid_recs.all():
        rank_matrix.data = recs_rnk[valid_recs]
        rank_matrix.indices = recommendations[valid_recs]
        rank_matrix.indptr = np.r_[0, np.cumsum(valid_recs.sum(axis=1))]
    else:
        rank_matrix.data = recs_rnk.ravel()
        rank_matrix.indices = recommendations.ravel()
        rank_matrix.indptr = np.arange(0, n_keys*topn+1, topn)

    hits_rank = eval_matrix.multiply(rank_matrix)
    # Note: scipy logical operations (OR, XOR, AND) are not supported yet
    # see https://github.com/scipy/scipy/pull/5411, using mult instead of AND
    return hits_rank