import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold, cross_val_score
from xgboost import XGBClassifier
import optuna

train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")

ids = test_data.pop("id")
encoder = LabelEncoder()
y = encoder.fit_transform(train_data.pop("Category"))

covariates = test_data.columns.intersection(train_data.columns)
train_data = train_data.loc[:, covariates]
test_data = test_data.loc[:, covariates]

def set_categories(data):
	categories = [not is_numeric_dtype(dtype) for dtype in data.dtypes]
	data.loc[:, categories] = data.loc[:, categories].astype("category")

set_categories(train_data)
set_categories(test_data)

splits = tuple(KFold(shuffle = True, random_state = 1234).split(train_data, y))

def model(n_estimators, lr, max_depth, min_child_weight):
	return XGBClassifier(enable_categorical = True, tree_method = "hist", objective = 'multi:softmax', eval_metric = 'error', n_estimators = n_estimators, learning_rate = lr, max_depth = max_depth, min_child_weight = min_child_weight, random_state = 1234)

def optimize():
	def objective(trial):
		n_estimators = trial.suggest_int('n_estimators', 10, 400)
		lr = trial.suggest_loguniform('lr', 0.01, 1)
		max_depth = trial.suggest_int('max_depth', 1, 60)
		min_child_weight = trial.suggest_int('min_child_weight', 1, 6)
		return np.mean(cross_val_score(model(n_estimators, lr, max_depth, min_child_weight), train_data, y, cv = splits))

	def saveBest(study, trial):
		if study.best_trial.number == trial.number:
			with open("best_params.txt", 'w') as myFile:
				myFile.write(str(trial.params))

	study = optuna.create_study(direction = "maximize")
	study.optimize(objective, callbacks = [saveBest])

def test(n_estimators, lr, max_depth, min_child_weight):
	print(np.mean(cross_val_score(model(n_estimators, lr, max_depth, min_child_weight), train_data, y, cv = splits)))

def export_submission(i, n_estimators, lr, max_depth, min_child_weight):
	predictions = encoder.inverse_transform(model(n_estimators, lr, max_depth, min_child_weight).fit(train_data, y).predict(test_data))
	pd.DataFrame({'id': ids, 'Expected': predictions}).to_csv(f"submission{i}.csv", index = False)

#optimize()
#test(302, 0.051666694958030716, 4, 6)
export_submission('', 302, 0.051666694958030716, 4, 6)
#export_submission(1, 255, 0.1078371086451241, 3, 6)
#export_submission(2, 256, 0.08774970448406516, 3, 6)
#export_submission(3, 277, 0.09602409888887524, 3, 6)
#export_submission(4, 286, 0.09569016227445425, 3, 6)
