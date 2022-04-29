from joblib import load

class Predictor:
    def __init__(self, model_name = 'stacked'):
        #if feature == 'tf-idf':
        #    pass
        #elif feature == 'word count':
        #    pass
        #else:
        #    raise Exception("Unknown feature extraction method! Must be either 'tf-idf' or 'word count'")
        
        self.vectorizer = load('extractors/tfidf_vectorizer.joblib')

        if model_name == 'stacked':
            self.model = load('models/stacked_classifier_rf_sv_mc_mlp_svr_lr.joblib')
        elif model_name == 'stacked2':
            self.model = load('models/stacked_classifier_rf_mc_lr.joblib')
        else:
            raise Exception("Unknown model selected! Choose stacked or stacked2")

    def predict(self, input):
        data = self.vectorizer.transform(input)
        return self.model.predict(data)