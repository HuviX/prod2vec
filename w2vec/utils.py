
from gensim.models.callbacks import CallbackAny2Vec
from gensim.models import Word2Vec
class callback(CallbackAny2Vec):
    """
    Callback to print loss after each epoch
    """
    def __init__(self):
        self.epoch = 0
        self.loss_history = []

    def on_epoch_end(self, model):
        loss = model.get_latest_training_loss()
        if self.epoch == 0:
            print('Loss after epoch {}: {}'.format(self.epoch, loss))
        else:
            print('Loss after epoch {}: {}'.format(self.epoch, loss - self.loss_previous_step))
        self.epoch += 1
        self.loss_previous_step = loss

def load_model(path):
    model = Word2Vec.load(path)
    model.init_sims(replace=True)
    return model

def mean(x):
    return sum(x)/len(x)


def get_mean_scores(scores):
    scoring = dict()
    for item, score in scores:
        try:
            scoring[item].append(score)
        except:
            scoring[item] = [score]
    for item, scoring_list in scoring.items():
        scoring[item] = mean(scoring_list)
    return [(k, v) for k, v in sorted(scoring.items(), key=lambda item: item[1], reverse=True)]


def get_prediction_for_session(view_model, cart_model, session: dict, n: int=10):
    views = session['views']
    to_cart = session['to_cart']
    res_cart = []
    res_view = []
    #топ N похожих векторов добавляются в список для каджого действия
    for vec in views:
        try:
            v = view_model.wv[vec]
            similar = view_model.wv.similar_by_vector(v, topn=n+1)[1:]
            res_view.append(similar)
        except:
            pass
            #print(vec, "not in vocab")
    for vec in to_cart:
        try:
            v = cart_model.wv[vec]
            similar = cart_model.wv.similar_by_vector(v, topn=n+1)[1:]
            res_cart.append(similar)
        except:
            pass
            #print(vec, "not in vocab")
    #Делается flat список. из [[1,2,3], [4, 5]] -> [1, 2, 3, 4, 5]
    res_view = [item for lis in res_view for item in lis]
    res_cart = [item for lis in res_cart for item in lis]
    #Объединение для списков для список
    res_cart += res_view
    #Сортим по скорам.
    res_cart.sort(key=lambda x: x[1], reverse=True)
    return [x[0] for x in res_cart[:n]]
