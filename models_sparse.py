import numpy as np
import pandas as pd
import scipy as sp
from polara.recommender.models import SVDModel
from polara.lib.sparse import inverse_permutation as inv_perm

SPARSE_MODE = True

try:
    from sksparse.cholmod import cholesky as cholesky_decomp_sparse
except ImportError:
    from scikit.sparse.cholmod import cholesky as cholesky_decomp_sparse
# there's a problem in cholmod - factor.solve_Lt returns inaccurate result
# github issue: https://github.com/scikit-sparse/scikit-sparse/issues/9
# have to use scipy's spsolve_triangular instead (added in scipy v.0.19)
def solve_triangular_sparse(x, y):
    return  sp.sparse.linalg.spsolve_triangular(x.L().T, y, lower=False)[inv_perm(x.P()), :]
    #return x.apply_Pt(x.solve_Lt(y))

def cholesky_factor_sparse(x):
    return x.L()[inv_perm(x.P()), :]
    #return x.apply_Pt(x.L())


class FeatureSimilarityMixin(object):
    def __init__(self, sim_mat, sim_idx, *args, **kwargs):        
        super(FeatureSimilarityMixin, self).__init__(*args, **kwargs)
        
        entities = [self.fields.userid, self.fields.itemid]
        self._sim_idx = {entity: pd.Series(index=idx, data=np.arange(len(idx)), copy=False)
                                 if idx is not None else None
                         for entity, idx in sim_idx.iteritems()
                         if entity in entities}
        self._sim_mat = {entity: mat for entity, mat in sim_mat.iteritems() if entity in entities}
        self._similarity = dict.fromkeys(entities)
        
        self._attach_model(self.on_change_event, self, '_clean_similarity')
        
    def _clean_similarity(self):
        self._similarity = dict.fromkeys(self._similarity.keys())
    
    @property
    def item_similarity(self):
        entity = self.fields.itemid
        return self.get_similarity_matrix(entity)

    @property
    def user_similarity(self):
        entity = self.fields.userid
        return self.get_similarity_matrix(entity)

    def get_similarity_matrix(self, entity):
        similarity = self._similarity.get(entity, None)
        if similarity is None:
            self._update_similarity(entity)
        return self._similarity[entity]

    def _update_similarity(self, entity):
        sim_mat = self._sim_mat[entity]
        if sim_mat is None:
            self._similarity[entity] = None
        else:
            if self.verbose:
                print 'Updating {} similarity matrix'.format(entity)
                
            entity_type = self.fields._fields[self.fields.index(entity)]
            index_data = getattr(self.index, entity_type)

            try: # check whether custom index is introduced
                entity_idx = index_data.training['old']
            except AttributeError: # fall back to standard case
                entity_idx = index_data['old']

            sim_idx = entity_idx.map(self._sim_idx[entity]).values
            sim_mat = self._sim_mat[entity][:, sim_idx][sim_idx, :]

            if sp.sparse.issparse(sim_mat):
                sim_mat.setdiag(1)
            else:
                np.fill_diagonal(sim_mat, 1)                
            self._similarity[entity] = sim_mat



class ColdSimilarityMixin(object):
    @property
    def cold_items_similarity(self):
        itemid = self.fields.itemid
        return self.get_cold_similarity(itemid)

    @property
    def cold_users_similarity(self):
        userid = self.fields.userid
        return self.get_cold_similarity(userid)

    def get_cold_similarity(self, entity):
        sim_mat = self._sim_mat[entity]
        
        if sim_mat is None:
            return None

        fields = self.fields
        entity_type = fields._fields[fields.index(entity)]
        index_data = getattr(self.index, entity_type)

        similarity_index = self._sim_idx[entity]
        seen_idx = index_data.training['old'].map(similarity_index).values
        cold_idx = index_data.cold_start['old'].map(similarity_index).values

        return sim_mat[:, seen_idx][cold_idx, :]


class CholeskyFactorsMixin(object):
    def __init__(self, *args, **kwargs):
        self._sparse_mode = SPARSE_MODE
        self.return_factors = True
        
        super(CholeskyFactorsMixin, self).__init__(*args, **kwargs)
        entities = [self.data.fields.userid, self.data.fields.itemid]
        self._cholesky  = dict.fromkeys(entities)
        
        self._features_weight = 0.999
        self.data._attach_model(self.data.on_change_event, self, '_clean_cholesky')

    def _clean_cholesky(self):
        self._cholesky = {entity:None for entity in self._cholesky.keys()}
        
    def _update_cholesky(self):
        for entity, cholesky in self._cholesky.iteritems():
            if cholesky is not None:
                self._update_cholesky_inplace(entity)

    @property
    def features_weight(self):
        return self._features_weight

    @features_weight.setter
    def features_weight(self, new_val):
        if new_val != self._features_weight:
            self._features_weight = new_val
            self._update_cholesky()
            self._renew_model()

    @property
    def item_cholesky_factor(self):
        itemid = self.data.fields.itemid
        return self.get_cholesky_factor(itemid)

    @property
    def user_cholesky_factor(self):
        userid = self.data.fields.userid
        return self.get_cholesky_factor(userid)
    
    def get_cholesky_factor(self, entity):
        cholesky = self._cholesky.get(entity, None)
        if cholesky is None:
            self._update_cholesky_factor(entity)
        return self._cholesky[entity]

    def _update_cholesky_factor(self, entity):
        entity_similarity = self.data.get_similarity_matrix(entity)
        if entity_similarity is None:
            self._cholesky[entity] = None
        else:
            if self._sparse_mode:
                cholesky_decomp = cholesky_decomp_sparse
                mode = 'sparse'
            else:
                raise NotImplementedError
                #entity_similarity = entity_similarity.toarray()
                #cholesky_decomp = cholesky_decomp_dense
                #mode = 'dense'
            
            weight = self.features_weight
            beta = (1.0 - weight) / weight
            if self.verbose:
                print 'Performing {} Cholesky decomposition for {} similarity'.format(mode, entity)
            self._cholesky[entity] = cholesky_decomp(entity_similarity, beta=beta)

    def _update_cholesky_inplace(self, entity):
        entity_similarity = self.data.get_similarity_matrix(entity)
        if self._sparse_mode:
            weight = self.features_weight
            beta = (1.0 - weight) / weight
            if self.verbose:
                print 'Updating Cholesky decomposition inplace for {} similarity'.format(entity)
            self._cholesky[entity].cholesky_inplace(entity_similarity, beta=beta)
        else:
            raise NotImplementedError


    def build(self, *args, **kwargs):            
        svd_matrix = self.get_training_matrix(dtype=np.float64)
        
        cholesky_users = self.user_cholesky_factor
        cholesky_items = self.item_cholesky_factor
        
        if self._sparse_mode:
            cholesky_factor = cholesky_factor_sparse
        else:
            raise NotImplementedError
        
        if cholesky_items is not None:
            svd_matrix = svd_matrix.dot(cholesky_factor(cholesky_items))

        if cholesky_users is not None:
            svd_matrix = cholesky_factor(cholesky_users).T.dot(svd_matrix)
            
        super(CholeskyFactorsMixin, self).build(*args, operator=svd_matrix, return_factors=self.return_factors, **kwargs)


class HybridSVD(CholeskyFactorsMixin, SVDModel):
    def __init__(self, *args, **kwargs):
        super(HybridSVD, self).__init__(*args, **kwargs)
        self.method = 'HybridSVD'
        self.return_factors = 'vh'
    
    def build(self, *args, **kwargs):
        super(HybridSVD, self).build(*args, **kwargs)
        
        if self._sparse_mode:
            cholesky_factor = cholesky_factor_sparse
            solve_triangular = solve_triangular_sparse
        else:
            raise NotImplementedError
        
        cholesky_items = self.item_cholesky_factor
        if cholesky_items is not None:
            v = self.factors[self.data.fields.itemid]
            self.factors['items_projector_left'] = solve_triangular(cholesky_items, v)
            self.factors['items_projector_right'] = cholesky_factor(cholesky_items).dot(v)

    def slice_recommendations(self, test_data, shape, start, stop, test_users=None):
        test_matrix, slice_data = self.get_test_matrix(test_data, shape, (start, stop))
        vr = self.factors['items_projector_right']
        vl = self.factors['items_projector_left']
        # projector is transposed
        scores = test_matrix.dot(vr).dot(vl.T)
        return scores, slice_data
        

class HybridSVDColdStart(CholeskyFactorsMixin, SVDModel):
    def __init__(self, *args, **kwargs):
        super(HybridSVDColdStart, self).__init__(*args, **kwargs)
        self.method = 'HybridSVD'
        self.return_factors = True
        self.filter_seen = False # there are no seen items in cold-start recommendations

    def get_recommendations(self):
        userid = self.data.fields.userid
        itemid = self.data.fields.itemid
        
        if self._sparse_mode:
            cholesky_factor = cholesky_factor_sparse
        else:
            cholesky_factor = cholesky_factor_dense
           
        cholesky_items = cholesky_factor(self.item_cholesky_factor)
        
        user_factors = self.factors[userid]
        #repr_users = self.data.representative_users
        #if repr_users is None:
        #    repr_users = self.data.index.userid.training
        #user_factors = user_factors[repr_users.new.values, :]
        
        item_factors = self.factors[itemid]
        s1 = np.reciprocal(self.factors['singular_values'])
        
        similarity_matrix = self.data.cold_items_similarity
        cold_factors = similarity_matrix.dot(cholesky_items.dot(item_factors) * s1[None, :])
        cold_factors /= np.linalg.norm(cold_factors, axis=0)[None, :]
        user_factors = user_factors / np.linalg.norm(user_factors, axis=0)[None, :]
        
        # computes cosine similarity between fake "cold" users and representative users
        scores = cold_factors.dot(user_factors.T)        
        top_similar_users = self.get_topk_elements(scores)
        
        #top_similar_idx = self.get_topk_elements(scores)
        #if self.data.representative_users is None:
        #    top_similar_users = top_similar_idx
        #else:
        #    top_similar_users = repr_users.new.values[top_similar_idx.ravel()].reshape(top_similar_idx.shape)
        return top_similar_users