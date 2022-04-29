from predictor.ml_processor import Predictor
from model_generator.model_training import *
import pandas as pd

def predictor_example():
    data = pd.read_excel('test_data/full_comment_list.xlsx', index_col=0)
    data.dropna(inplace=True, subset=['translated_string'])
    model = Predictor()
    sentiment = model.predict(data['translated_string'])
    data['sentiment'] = sentiment
    data.to_excel('test_data/output_comment_list.xlsx')


def training_example():
    data = pd.read_excel('test_data/labeled_data.xlsx', index_col=0)
    data.dropna(inplace=True, subset=['sentiment (-1, 0, 1)', 'original_comment', 'en_translation'])
    
    X = data['en_translation'] # used as the input to the ML algorithm
    Y = data['sentiment (-1, 0, 1)'] # used as the output of the ML algorithm
    X = X.astype(str) # probably not needed, because we removed all unvalid values

    # simple_logistic_regression(X, Y)
    # simple_random_forest(X, Y)
    # simple_support_vector_machine(X, Y)
    # simple_k_nearest_neighbors(X, Y)
    # simple_multy_layer_perceptron(X, Y)



if __name__ == '__main__':
    training_example()

