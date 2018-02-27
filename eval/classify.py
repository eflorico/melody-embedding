import time
import itertools
import random
import numpy as np
from sklearn import svm
#from sklearn.model_selection import KFold, GridSearchCV
from sklearn.cross_validation import KFold
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import PCA

# Timing
def tick(label=None):
    t1 = time.time()

    if tick.t0 != None:
        print("%.4fs" % (t1 - tick.t0))

    if label != None:
        print(label)

    tick.t0 = t1
tick.t0 = None

# Read data
data = np.load('data.npy')
mids = data[:, 0]
fids = data[:, 1]
label_probs = data[:, 2:17]
features = data[:, 17:]

# Prepare data
labels_num = np.argmax(label_probs, axis=1).reshape(-1, 1)
#enc = OneHotEncoder(sparse=False)
#labels_oh = enc.fit_transform(labels_num)

# Zero-center, unit variance
features = (features - features.mean(axis=0)) / features.std(axis=0)

# Group by song
# tick("Grouping...")
# song_features = {}
# song_labels = {}
# for fid in fids:
# 	if fid not in song_features:
# 		positions = (fids == fid)
# 		song_features[fid] = features[positions]
# 		song_labels[fid] = labels_num[positions, 0][0]
# song_ids = song_labels.keys()

# Song mean
# tick('Grouping...')
# unique_fids = np.unique(fids)
# features_mean = np.zeros((len(unique_fids), features.shape[1]))
# labels_mean = np.zeros(len(unique_fids))
# for i, fid in enumerate(unique_fids):
# 	positions = (fids == fid)
# 	features_mean[i] = np.mean(features[positions], axis=0)
# 	labels_mean[i] = labels_num[positions, 0][0]
# features = features_mean
# labels = labels_mean

# Subset for grid search
#random.seed(0)
#subset = list(range(0, len(features)))
#subset = random.sample(subset, 1000)
#features = features[subset, :]
#labels_num = labels_num[subset]
#labels_oh = labels_oh[subset, :]


# k Fold
def test_weighted(model):
	print(', '.join('%s: %s' % (k,v) for k, v in model.get_params().items()))
	print("%6s\t%6s\t%6s\t%6s" % ('Acc', 'Prec', 'Rec', 'F1'))

	kf = KFold(n=len(song_ids), n_folds=10, shuffle=True)

	scores = []
	for train, test in kf:
		try:
			train_features = np.concatenate([ song_features[song_ids[i]] for i in train ])
			train_labels = np.concatenate([ 
				[ song_labels[song_ids[i]] ] * len(song_features[song_ids[i]]) 
				for i in train ])
			model.fit(train_features, train_labels)

			test_features = np.concatenate([ song_features[song_ids[i]] for i in test ])
			y_true = [ song_labels[song_ids[i]] for i in test ]

			# Weighted voting
			y_pred_prob = np.array(clf.predict_proba(test_features))
			y_pred_aggr = np.zeros((len(test), 15))

			positions = np.cumsum([0] + [ len(song_features[song_ids[i]]) for i in test ])
			for i in xrange(len(test)):
				y_pred_aggr[i, :] = np.sum(y_pred_prob[positions[i]:positions[i+1], :], axis=0)
			y_pred = np.argmax(y_pred_aggr, axis=1).reshape(-1, 1)

			acc = accuracy_score(y_true, y_pred)
			prec = precision_score(y_true, y_pred, average='weighted')
			rec = recall_score(y_true, y_pred, average='weighted')
			f1 = f1_score(y_true, y_pred, average='weighted')
			scores.append([ acc, prec, rec, f1 ])

			print("%.4f\t%.4f\t%.4f\t%.4f" % (acc, prec, rec, f1))
		except KeyboardInterrupt:
			print("Aborted")
			return

	if len(scores) > 0:
		print("Mean scores:")
		n = len(scores)
		m = len(scores[0])
		means = [ 
			sum(s[i] for s in scores) / n
			for i in range(0, m) 
		]
		print(("%.4f\t" * m) % tuple(means))

def test(model):
	print(', '.join('%s: %s' % (k,v) for k, v in model.get_params().items()))
	print("%6s\t%6s\t%6s\t%6s" % ('Acc', 'Prec', 'Rec', 'F1'))

	kf = KFold(n=len(features), n_folds=10, shuffle=True)

	scores = []
	for train, test in kf:
		try:
			model.fit(features[train], labels_num[train].ravel())

			y_true = labels_num[test]
			y_pred_prob = np.array(clf.predict_proba(features[test]))
			y_pred = np.argmax(y_pred_prob, axis=1).reshape(-1, 1)

			acc = accuracy_score(y_true, y_pred)
			prec = precision_score(y_true, y_pred, average='weighted')
			rec = recall_score(y_true, y_pred, average='weighted')
			f1 = f1_score(y_true, y_pred, average='weighted')
			scores.append([ acc, prec, rec, f1 ])

			print("%.4f\t%.4f\t%.4f\t%.4f" % (acc, prec, rec, f1))
		except KeyboardInterrupt:
			print("Aborted")
			return

	if len(scores) > 0:
		print("Mean scores:")
		n = len(scores)
		m = len(scores[0])
		means = [ 
			sum(s[i] for s in scores) / n
			for i in range(0, m) 
		]
		print(("%.4f\t" * m) % tuple(means))

# tick("Random Forest Gini...")
# params = { 'n_estimators': 256, 'n_jobs': 16, 'criterion': 'gini' }
# clf = RandomForestClassifier(**params)
# test(clf)
# tick()

# tick("Random Forest Entropy...")
# params = { 'n_estimators': 256, 'n_jobs': 16, 'criterion': 'entropy' }
# clf = RandomForestClassifier(**params)
# test(clf)
# tick()

tick("Lin Reg...")
params = { 'n_jobs': -1, 'C': 0.001 }
clf = LogisticRegression(**params)
test(clf)

tick("kNN")
clf = KNeighborsClassifier(n_jobs=-1)
test(clf)

# tick("kNN grid search...")
# param_grid = [
# 	{ 'n_neighbors': [3,5,10,20,50], 'n_jobs':[16] }
# ]
# clf = GridSearchCV(KNeighborsClassifier(), param_grid)
# clf.fit(features, labels_num.ravel())
# print(clf.best_estimator_)


# tick("AdaBoost")
# clf = AdaBoostClassifier(n_estimators=64)
# test(clf)

# tick("RF grid search...")
# param_grid = [
# 	{ 'n_estimators': [ 64, 128, 256 ], 'n_jobs': [16] },
# ]
# clf = GridSearchCV(RandomForestClassifier(), param_grid)
# clf.fit(features, labels_num.ravel())
# print(clf.best_estimator_)


# tick("AdaBoost grid search...")
# param_grid = [
# 	{ 'n_estimators': [ 64, 128, 256 ] }
# ]
# clf = GridSearchCV(AdaBoostClassifier(), param_grid)
# clf.fit(features, labels_num.ravel())
# print(clf.best_estimator_)

# # PCA
# tick("PCA...")
# pca = PCA(n_components=10)
# features = pca.fit_transform(features)

# tick("Logreg grid search...")
# param_grid = [
# 	{ 'C': [10**i for i in range(-3,0)], 'n_jobs': [16] }
#  ]
# clf = GridSearchCV(LogisticRegression(), param_grid)
# clf.fit(features, labels_num.ravel())
# print(clf.best_estimator_)

# tick("SVM grid search...")
# param_grid = [
# 	{ 'C': [10**i for i in range(3,8)], 'kernel': ['linear'] },
# 	{ 'C': [ 10 ** i for i in range(3, 8) ], 
# 	 	'gamma': [ 2 ** i for i in range(-10, -4) ], 
# 	 	'kernel': ['rbf']
# 	 },
# 	{'C': [10**i for i in range(3,8)], 'degree': [3, 4], 'kernel': ['poly'] },
#  ]
# clf = GridSearchCV(svm.SVC(), param_grid)
# clf.fit(features, labels_num.ravel())
# print(clf.best_estimator_)


# tick("SVM grid search...")
# params = { 'C': 2 ** 6, 'gamma': 2 ** -9, 'kernel': 'rbf' }
# param_grid = [
# #	{ 'kernel':['rbf']}
#    { 'C': [10**i for i in range(3,8)], 'kernel': ['linear']},
#    { 'C': [2 ** 6], 'gamma': [2 ** -9], 'kernel': ['rbf'] }
# #  { 'C': [ 10 ** i for i in range(4, 9) ], 
# #  	'gamma': [ 2 ** i for i in range(-10, -4) ], 
# #  	'kernel': ['rbf']
# #  },
# # {'C': [1000, 10000], 'degree': [3, 4], 'kernel': ['poly']},
#  ]

# #train, test = KFold(n_splits=100, shuffle=True)

# clf = GridSearchCV(svm.SVC(), param_grid)
# clf.fit(features[subset], labels[subset])
# print(clf.best_estimator_)

# tick("SVM...")
# params = { 'kernel': 'rbf' }
# test(clf.best_estimator_)

tick()