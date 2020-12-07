import scipy.sparse as sp
import pickle
import numpy as np
from gensim.models import Word2Vec
from .utils import get_scores, mean


class cosine_model():
    def __init__(self, path):
        with open(path+'/old2new_dict.pkl', 'rb') as handle:
            self.old2new = pickle.load(handle)
        with open(path+'/new2old_dict.pkl', 'rb') as handle:
            self.new2old = pickle.load(handle)
        self.similarities_view = sp.load_npz(path+'/similarities_view.npz')
        self.similarities_view = self.similarities_view.tocsc()
        
        self.similarities_cart_add = sp.load_npz(path+'/similarities_cart_add.npz')
        self.similarities_cart_add = self.similarities_cart_add.tocsc()


    def _predict(self, session, topn, how='mean'):
        res_view = []
        res_cart = []
        fancy = []
        for vec in session['views']:
            new_item_id = self.old2new[vec]
            fancy.append(new_item_id)
        col = self.similarities_view[:, fancy]
        try:
            ix = np.argpartition(col.data,kth=-topn-1, axis=0)[-topn-1:]
            indices = col.indices[ix]
            values = col.data[ix]
        except ValueError:
            indices = col.indices
            values = col.data
        for i, ind in enumerate(indices):
            res_view.append([self.new2old[ind], values[i]])
        fancy = []
        for vec in session['to_cart']:
            new_item_id = self.old2new[vec]
            fancy.append(new_item_id)
        col = self.similarities_cart_add[:, fancy]
        try:
            ix = np.argpartition(col.data,kth=-topn-1, axis=0)[-topn-1:]
            indices = col.indices[ix]
            values = col.data[ix]
        except ValueError:
            indices = col.indices
            values = col.data

        for i, ind in enumerate(indices):
            res_cart.append([self.new2old[ind], values[i]])

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
                v = self.model.wv[vec]
                similar = self.model.wv.similar_by_vector(v, topn=n+1,)[1:]
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
    def __init__(self, path='data/top50.txt'):
        self.top50 = []
        with open(path) as f:
            a = f.readline()
            while a:
                self.top50.append(a.strip())
                a = f.readline()
        self.top50 = self.top50[:50]


    def get_prediction(self):
        return self.top50
