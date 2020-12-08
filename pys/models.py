import scipy.sparse as sp
import pickle
import numpy as np
from gensim.models import Word2Vec
from .utils import get_scores, mean


class cosine_model():
    def __init__(self, path):
        self.similarities_view = sp.load_npz(path+'/similarities_view.npz')
        self.similarities_view = self.similarities_view.tocsc()
        
        self.similarities_cart_add = sp.load_npz(path+'/similarities_cart_add.npz')
        self.similarities_cart_add = self.similarities_cart_add.tocsc()


    def _predict(self, session, topn, how='mean'):
        res_view = []
        res_cart = []
        for vec in session['views']:
            new_item_id = vec
            col = self.similarities_view[:, new_item_id]
            try:
                ix = np.argpartition(col.data,kth=-topn-1, axis=0)[-topn-1:]
                indices = col.indices[ix]
                values = col.data[ix]
            except ValueError:
                indices = col.indices
                values = col.data
            for i, ind in enumerate(indices):
                res_view.append([ind, values[i]])

        for vec in session['to_cart']:
            new_item_id = vec
            col = self.similarities_cart_add[:, new_item_id]
            try:
                ix = np.argpartition(col.data,kth=-topn-1, axis=0)[-topn-1:]
                indices = col.indices[ix]
                values = col.data[ix]
            except ValueError:
                indices = col.indices
                values = col.data

            for i, ind in enumerate(indices):
                res_cart.append([ind, values[i]])

        res_cart += res_view
        #Сортим по скорам.
        res_cart.sort(key=lambda x: x[1], reverse=True)

        return get_scores(res_cart, how)[:topn]


class prod2Vec:
    def __init__(self, path):
        self.model = Word2Vec.load(path)
        self.model.init_sims(replace=True)


    def _predict(self, session, n):
        prediction = []
        for vec in session:
            try:
                v = self.model.wv[str(vec)]
                similar = self.model.wv.similar_by_vector(v, topn=n+1,)[1:]
                similar = [(int(x[0]), x[1]) for x in similar]
                prediction.append(similar)
            except KeyError:
                continue
        prediction = [item for lis in prediction for item in lis]
        return prediction


    def get_prediction_for_session(self, session: dict, topn: int, how: str):
        views = session['views']
        to_cart = session['to_cart']
        res_view = self.model._predict(views, topn)
        res_cart = self.model._predict(to_cart, topn)
        res_cart += res_view
        #Сортим по скорам.
        res_cart.sort(key=lambda x: x[1], reverse=True)
        return get_scores(res_cart, how)[:topn]


class CombinedProd2Vec:
    def __init__(self, paths):
        self._view_model = prod2Vec(paths[0])
        self._cart_model = prod2Vec(paths[1])
    

    def get_prediction_for_session(self, session: dict, topn: int, how: str):
        views = session['views']
        to_cart = session['to_cart']
        res_view = self._view_model._predict(views, topn)
        res_cart = self._cart_model._predict(to_cart, topn)
        res_cart += res_view
        #Сортим по скорам.
        res_cart.sort(key=lambda x: x[1], reverse=True)
        return get_scores(res_cart, how)[:topn]


class top():
    def __init__(self, path='data/top50.pkl'):
        with open(path, 'rb') as handle:
            self.top50 = pickle.load(handle) 


    def get_prediction(self):
        return self.top50
