from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import preprocessing
from scipy.stats import loguniform
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import StackingClassifier

from joblib import dump, load

def count_vector_transform(X, Y, test_size = 0.3):
    cv = CountVectorizer() # feature extraction
    X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=test_size) # train/test split

    ctmTr = cv.fit_transform(X_train) # train and apply transform
    X_test_dtm = cv.transform(X_test) # apply transform

    return cv, ctmTr, X_test_dtm, Y_train, Y_test

def tfidf_transform(X, Y, test_size = 0.3):
    cv = TfidfVectorizer() # feature extraction
    X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=test_size) # train/test split

    ctmTr = cv.fit_transform(X_train) # train and apply transform
    X_test_dtm = cv.transform(X_test) # apply transform

    return cv, ctmTr, X_test_dtm, Y_train, Y_test


def simple_logistic_regression(X, Y, max_iter = 300):

    cv, ctmTr, X_test_dtm, Y_train, Y_test = count_vector_transform(X, Y)
    
    lr = LogisticRegression(max_iter = max_iter)

    lr.fit(ctmTr, Y_train)
    lr_score = lr.score(X_test_dtm, Y_test)

    print("Results for Logistic Regression with CountVectorizer")
    print(lr_score)

    dump(lr, 'models/simple_logistic_regression.joblib')
    dump(cv, 'extractors/lr_word_count.joblib') 

def simple_random_forest(X, Y, n_estimators = 200):
    
    cv, ctmTr, X_test_dtm, Y_train, Y_test = count_vector_transform(X, Y)

    text_classifier = RandomForestClassifier(n_estimators=n_estimators)

    text_classifier.fit(ctmTr, Y_train)
    rnn_score = text_classifier.score(X_test_dtm, Y_test)

    print("Results for Random Forest Classifier with CountVectorizer")
    print(rnn_score)
    
    dump(text_classifier, 'models/simple_random_forest.joblib')
    dump(cv, 'extractors/rf_word_count.joblib') 


def simple_support_vector_machine(X, Y):
    
    cv, ctmTr, X_test_dtm, Y_train, Y_test = count_vector_transform(X, Y)

    svcl = svm.SVC()

    svcl.fit(ctmTr, Y_train)
    svcl_score = svcl.score(X_test_dtm, Y_test)
    
    print("Results for Support Vector Machine with CountVectorizer")
    print(svcl_score)

    dump(svcl, 'models/simple_support_vector_machine.joblib')
    dump(cv, 'extractors/svm_word_count.joblib') 


def simple_k_nearest_neighbors(X, Y, n_neighbors = 5):

    cv, ctmTr, X_test_dtm, Y_train, Y_test = count_vector_transform(X, Y)

    knn = KNeighborsClassifier(n_neighbors=5)

    knn.fit(ctmTr, Y_train)
    knn_score = knn.score(X_test_dtm, Y_test)

    print("Results for KNN Classifier with CountVectorizer")
    print(knn_score)

    dump(knn, 'models/simple_k_nearest_neighbors.joblib')
    dump(cv, 'extractors/knn_word_count.joblib') 

def simple_multy_layer_perceptron(X, Y, max_iter = 300):

    cv, ctmTr, X_test_dtm, Y_train, Y_test = count_vector_transform(X, Y)

    clf = MLPClassifier(random_state=1, max_iter=300)

    # Accuracy score
    clf.fit(ctmTr, Y_train)
    mlp_score = clf.score(X_test_dtm, Y_test)
    
    print("Results for MLP Classifier with CountVectorizer")
    print(mlp_score)

    dump(clf, 'models/simple_multy_layer_perceptron.joblib')
    dump(cv, 'extractors/mlp_word_count.joblib') 








