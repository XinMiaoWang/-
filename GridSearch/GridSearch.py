from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import lightgbm as lgb
from sklearn.datasets import load_iris
import warnings

warnings.filterwarnings(action='ignore', category=DeprecationWarning)

iris = load_iris() # load iris dataset from sklearn
X_train, X_test, y_train, y_test = train_test_split( iris.data, iris.target, random_state=0 )

# --------------- gridsearch start ---------------
parameters = {
        'max_depth': [ 5, 7, 9, 11, 13, 15],
        'num_leaves': [ 10, 15, 20, 25, 30, 35, 40, 45, 50]
    }

estimator = lgb.LGBMClassifier(
                objective = 'multiclass',
                learning_rate = 0.1,
                n_estimators = 1000
            )

# 5 folds
gsearch = GridSearchCV( estimator , param_grid = parameters, scoring='accuracy', cv=5 )
gsearch.fit( X_train, y_train )

print("Each score: ", gsearch.grid_scores_)
print("Best score: ", gsearch.best_score_)
print("Best parameters: ", gsearch.best_params_)
# --------------- gridsearch end ---------------