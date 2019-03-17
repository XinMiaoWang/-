from sklearn.model_selection import train_test_split
import lightgbm as lgb
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
import warnings

warnings.filterwarnings(action='ignore', category=DeprecationWarning)

iris = load_iris() # load iris dataset from sklearn
X_train, X_test, y_train, y_test = train_test_split( iris.data, iris.target, random_state=0 )

# LightGBM Model
gbm = lgb.LGBMClassifier(
                objective = 'multiclass',
                learning_rate = 0.1,
                n_estimators = 1000,
                max_depth = 5,
                num_leaves = 10
            )

gbm.fit(X_train, y_train)
y_pred = gbm.predict(X_test)
result = accuracy_score(y_test, y_pred)

print('Accuracy score : ', result)

