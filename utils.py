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

def mean(x):
    return sum(x)/len(x)


def get_scores(scores, how: str = 'mean'):
    """
    
    """
    scoring = dict()
    for item, score in scores:
        try:
            scoring[item].append(score)
        except:
            scoring[item] = [score]
    for item, scoring_list in scoring.items():
        scoring[item] = sum(scoring_list) if how == 'sum' else mean(scoring_list)
    return [(k, v) for k, v in sorted(scoring.items(), key=lambda item: item[1], reverse=True)]
