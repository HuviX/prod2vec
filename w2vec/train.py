import gc
import pickle
import gensim
from gensim.models import Word2Vec
from utils import callback
import matplotlib.pyplot as plt
import argparse
import os


def load_data(path: str):
    with open('train_df/'+path, 'rb') as f:
        data = pickle.load(f)
    return data


def init_model(window_len: int, min_count: int)-> Word2Vec():
    model = Word2Vec(window=window_len, sg=1, hs=0, 
                 negative=10, size=128, workers=32,
                 alpha=0.01, min_alpha=0.0001, min_count=min_count,
                 seed=14, compute_loss=True)

    return model


def main(n_epochs: int, min_count: int):
    paths = ['cart_sessions.pkl', 'view_sessions.pkl']#, 'combined_sessions.pkl']
    model_names = ['cart', 'view']#, 'combined']
    for model_name, path in zip(model_names, paths):
        data = load_data(path)

        max_len = max(len(x) for x in data)
        if model_name == 'cart':
            model = init_model(max_len, min_count//2)
        else:
            model = init_model(max_len, min_count)
        #train model
        model.build_vocab(data, progress_per=200)
        model.train(data, total_examples=model.corpus_count, 
                        epochs=n_epochs, report_delay=1, compute_loss=True, 
                        callbacks=[callback()])

        model.save(f"weights/{model_name}.model")
    print("Done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_epochs', help='num_of_epochs', default=30)
    parser.add_argument('--freq', help='min count of one word', default=5)
    args = parser.parse_args()
    n_epochs = int(args.n_epochs)
    min_count = int(args.freq)
    if not os.path.exists('weights'):
        os.makedirs('weights')
    main(n_epochs, min_count)
