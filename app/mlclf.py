#!/usr/bin/env python
import argparse
from collections import Counter, defaultdict
from datetime import datetime
import logging
import numpy as np
import pandas as pd


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
#TODO from sklearn.metrics import f1_score 
from sklearn.model_selection import cross_val_score, cross_validate, GridSearchCV, StratifiedKFold, train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from sklearn.svm import LinearSVC

import sys
import time

from utils import multilabel_col, reverse_diag


def main():

	# ================
	# time managment #
	# ================

	program_st = time.time()
	clf_durations = defaultdict(list)

	# =======================
	# predefined parameters #
	# =======================

	n_jobs = args.n_jobs
	cv = 10
	cv_dict = {}
	vectorizer = TfidfVectorizer(lowercase=args.lowercase,
								 max_features=args.max_features,
								 ngram_range=(1,1),
								 stop_words=None)

	
	# ================================
	# classification logging handler #
	# ================================
	logging_filename = f"../logs/mlclf_{args.corpus_name}.log"
	logging.basicConfig(level=logging.DEBUG, filename=logging_filename, filemode="w")
	console = logging.StreamHandler()
	console.setLevel(logging.INFO)
	formatter = logging.Formatter("%(levelname)s: %(message)s")
	console.setFormatter(formatter)
	logging.getLogger('').addHandler(console)

	mpl_logger = logging.getLogger('matplotlib')
	mpl_logger.setLevel(logging.WARNING)

	
	# =================
	# corpus loading  # 
	# =================

	if args.corpus_name == "poet":
		corpus = pd.read_csv("../corpora/german_modern_poems_epochpoet.csv")
	elif args.corpus_name == "year":
		corpus = pd.read_csv("../corpora/german_modern_poems_epochyear.csv")
	else:
		logging.warning(f"Couldn't find a corpus with the name '{args.corpus_name}'.")
	
	class_name1 = "epoch_year"
	class_name2 = "epoch_poet"
	text_name = "poem"


	# TODO: vllt weg?

	# extracting class weights #
	class1_counts = dict(Counter(class1))
	class2_counts = dict(Counter(class2))



	class1_weights = {"Expressionismus": class1_counts["Expressionismus"],
					 "Jahrhundertwende": class1_counts["Jahrhundertwende"],
					 "Naturalismus": class1_counts["Naturalismus"]}

	class2_weights = {"Expressionismus": class2_counts["Expressionismus"],
					 "Jahrhundertwende": class2_counts["Jahrhundertwende"],
					 "Naturalismus": class2_counts["Naturalismus"]}

	if args.multilabel:
		cols = ["epoch_year", "epoch_poet"]
		corpus["epochs"] = corpus[cols].apply(lambda row: multilabel_col(list(row.values)), axis=1) 
		class3 = [set(row['epochs'].split(',')) for index, row in corpus[["epochs"]].iterrows()]
		class3 = MultiLabelBinarizer().fit_transform(class3)

	logging.info(f"Read {args.corpus_name} corpus ({int((time.time() - program_st)/60)} minute(s)).")
	
	# ================
	# classification # 
	# ================

	features = corpus[text_name]
	class1 = corpus[class_name1]
	class2 = corpus[class_name2]

	

	# ============
	# Linear SVM #
	# ============

	lsvm_st = time.time()
	lsvm_pipe = Pipeline(steps=[("vect", vectorizer),
								("clf", LinearSVC())])

	lsvm_parameters = {"clf__penalty": ["l2"],
					   "clf__loss": ["squared_hinge"],
					   "clf__tol": [1e-6, 1e-5, 1e-4, 1e-3],
					   "clf__C": list(range(1, 11)),
					   "clf__max_iter": [100, 500, 1000, 2000, 3000, 5000],
					   "clf__class_weight": [None, "balanced"]}
	#ALTERNATIVE
	"""
	lsvm_parameters = {"clf__penalty": ["l2"],
					   "clf__loss": ["squared_hinge"],
					   "clf__tol": [1e-3],
					   "clf__C": [1.0],
					   "clf__max_iter": [100]}
	"""

	#TODO: lsvm_parameters.update({"clf__class_weight": [class1_weights]})
	lsvm_grid1 = GridSearchCV(lsvm_pipe, 
							 lsvm_parameters,
							 cv=cv, 
							 error_score=0.0,
							 n_jobs=args.n_jobs,
							 scoring="f1_macro")

	#TODO: lsvm_parameters.update({"clf__class_weight": [class2_weights]})
	lsvm_grid2 = GridSearchCV(lsvm_pipe, 
							 lsvm_parameters,
							 cv=cv, 
							 error_score=0.0,
							 n_jobs=args.n_jobs,
							 scoring="f1_macro")


	lsvm_cv_scores1 = cross_validate(lsvm_grid1,
									 features, 
									 class1, 
									 cv=cv, 
									 return_estimator=False,
									 scoring="f1_macro")


	lsvm_cv_scores2 = cross_validate(lsvm_grid2, 
									  features, 
									  class2, 
									  cv=cv, 
									  return_estimator=False,
									  scoring="f1_macro")

	
	if args.multilabel:
		lsvm_pipe = Pipeline(steps=[("vect", vectorizer),
									("clf", OneVsRestClassifier(LinearSVC()))])

		lsvm_parameters = {"clf__estimator__penalty": ["l2"],
						   "clf__estimator__loss": ["squared_hinge"],
						   "clf__estimator__tol": [1e-6, 1e-5, 1e-4, 1e-3],
						   "clf__estimator__C": list(range(1, 11)),
						   "clf__estimator__max_iter": [100, 500, 1000, 2000, 3000, 5000]}

		lsvm_grid3 = GridSearchCV(lsvm_pipe, 
								 lsvm_parameters,
								 cv=cv, 
								 error_score=0.0,
								 n_jobs=args.n_jobs,
								 scoring="f1_macro")


		lsvm_cv_scores3 = cross_validate(lsvm_grid3, 
										 features, 
										 class3, 
										 cv=cv,
										 return_estimator=False,
										 scoring="f1_macro")
		

	
	cv_dict["LSVM"] = {"year": np.mean(lsvm_cv_scores1["test_score"]),
					   "poet": np.mean(lsvm_cv_scores2["test_score"]),
					   "multilabel": np.mean(lsvm_cv_scores3["test_score"])}


	lsvm_duration = float(time.time() - lsvm_st)
	clf_durations["LSVM"].append(lsvm_duration)
	logging.info(f"Run-time LSVM: {lsvm_duration} seconds")


	# =====================
	# Logistic Regression #
	# =====================


	lr_st = time.time()

	lr_pipe = Pipeline(steps=[("vect", vectorizer),
							  ("clf", LogisticRegression())])


	


	lr_parameters = {"clf__penalty": ["l1", "l2"],
					 "clf__tol": [1e-5, 1e-3],
					 "clf__C": list(range(1, 11)),
					 "clf__solver": ["liblinear"],
					 "clf__max_iter": [1000, 3000, 5000],
					 "clf__class_weight": [None, "balanced"]}

	#ALTERNATIVE
	"""
	lr_parameters = {"clf__penalty": ["l1"],
					 "clf__solver": ["liblinear"],
					 "clf__max_iter": [1000]}
	"""

	#TODO: lr_parameters.update({"clf__class_weight": [class1_weights]})
	lr_grid1 = GridSearchCV(lr_pipe, 
							lr_parameters,
							cv=cv, 
							error_score=0.0,
							n_jobs=args.n_jobs,
							scoring="f1_macro")

	#TODO: lr_parameters.update({"clf__class_weight": [class2_weights]})
	lr_grid2 = GridSearchCV(lr_pipe, 
							lr_parameters,
							cv=cv, 
							error_score=0.0,
							n_jobs=args.n_jobs,
							scoring="f1_macro")


	lr_cv_scores1 = cross_validate(lr_grid1,
								   features, 
								   class1, 
								   cv=cv, 
								   return_estimator=False,
								   scoring="f1_macro")


	lr_cv_scores2 = cross_validate(lr_grid2, 
								   features, 
								   class2, 
								   cv=cv, 
								   return_estimator=False,
								   scoring="f1_macro")


	if args.multilabel:
		# class weights won't be used here
		lr_pipe = Pipeline(steps=[("vect", vectorizer),
							  	  ("clf", OneVsRestClassifier(LogisticRegression()))])

		lr_parameters = {"clf__estimator__penalty": ["l1", "l2"],
						 "clf__estimator__tol": [1e-5, 1e-3],
						 "clf__estimator__C": list(range(1, 11)),
						 "clf__estimator__solver": ["liblinear"],
						 "clf__estimator__max_iter": [1000, 3000, 5000]}

		lr_grid3 = GridSearchCV(lr_pipe, 
							   lr_parameters,
							   cv=cv, 
							   error_score=0.0,
							   n_jobs=args.n_jobs,
							   scoring="f1_macro")


		lr_cv_scores3 = cross_validate(lr_grid3,
									   features, 
									   class3, 
									   cv=cv, 
									   return_estimator=False,
									   scoring="f1_macro")

	
	cv_dict["LR"] = {"year": np.mean(lr_cv_scores1["test_score"]),
					 "poet": np.mean(lr_cv_scores2["test_score"]),
					 "multilabel": np.mean(lr_cv_scores3["test_score"])}


	lr_duration = float(time.time() - lr_st)
	clf_durations["LR"].append(lr_duration)
	logging.info(f"Run-time LR: {lr_duration} seconds")

	# ===========================================
	# Saving classification results & durations #
	# ===========================================

	cv_df = pd.DataFrame(cv_dict)
	cv_name = f"mlcv_{args.corpus_name}"
	if args.save_date:
		cv_name += f"({datetime.now():%d.%m.%y}_{datetime.now():%H:%M})"
	cv_df.to_csv(f"../results/{cv_name}.csv")


	durations_df = pd.DataFrame(clf_durations.items(), columns=["clf", "durations"])
	durations_name = f"mldurations_{args.corpus_name}"
	if args.save_date:
		durations_name += f"({datetime.now():%d.%m.%y}_{datetime.now():%H:%M})"
	durations_df.to_csv(f"../results/{durations_name}.csv")


	program_duration = float(time.time() - program_st)
	logging.info(f"Run-time: {int(program_duration)/60} minute(s).")
	
if __name__ == "__main__":
	
	parser = argparse.ArgumentParser(prog="mlclf", description="Comparison of LSVM and LR classification.")
	parser.add_argument("--corpus_name", "-cn", type=str, default="year", help="Indicates the corpus. Default is 'year'. Another possible value is 'poet'.")
	parser.add_argument("--lowercase", "-l", type=bool, default=False, help="Indicates if words should be lowercased.")
	parser.add_argument("--max_features", "-mf", type=int, default=60000, help="Indicates the number of most frequent words.")
	parser.add_argument("--multilabel", "-ml", action="store_true", help="Indicates if multilabel classification should be used.")
	parser.add_argument("--n_jobs", "-nj", type=int, default=1, help="Indicates the number of processors used for computation.")
	parser.add_argument("--save_date", "-sd", action="store_true", help="Indicates if the creation date of the results should be saved.")
	
	args = parser.parse_args()

	main()
