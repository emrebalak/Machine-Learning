import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn import tree
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
import csv

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


# Predict best hyperparameters values for decision tree
def predict_best_value_for_decision_tree():
    clf = GridSearchCV(tree.DecisionTreeClassifier())
    clf.fit(X=extracted_df, y=y)
    tree_model = clf.best_estimator_
    print(clf.best_score_, clf.best_params_)


def k_fold():
    scores = cross_val_score(tree.DecisionTreeRegressor(), extracted_df, y, cv=5)
    print("Cross-validated scoresscores", scores)


def load_data(filename):
    return pd.read_csv(filename)


def pre_processing(df, number_of_feature, is_train_model=False):
    df = df.drop(['X3', 'X31', 'X32', 'X127', 'X128', 'X590'], axis=1)

    # seperate label and features
    x = df.iloc[:, 0:588].values

    # Standardizing the features
    x = StandardScaler().fit_transform(x)

    y = None
    if is_train_model:
        y = df.iloc[:, 589]

    # PCA
    pca = PCA(n_components=number_of_feature)
    extracted_features = pca.fit_transform(x)
    extracted_df = pd.DataFrame(data=extracted_features)
    return extracted_df, y


def train_model(df, y, test_df):
    model = tree.DecisionTreeRegressor()
    model.fit(df, y)
    return model.predict(test_df)


def write_output(predicted_list):
    with open('submission.csv', mode='w') as predicted_file:
        submission = csv.writer(predicted_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        submission.writerow(['ID', 'Predicted'])
        a = 1
        for i in predicted_list:
            submission.writerow([str(a), int(i)])
            a = a + 1


train_df = load_data('train.csv')
test_df = load_data('test.csv')


extracted_df, y = pre_processing(train_df, 10, is_train_model=True)
test_df, y_none = pre_processing(test_df, 10)

predicted_list = train_model(extracted_df, y, test_df)

write_output(predicted_list)











