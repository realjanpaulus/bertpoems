## TODO:
# - alles überprüfen
# - unnötiges rauscutten
# - njobs bei grid search mal testen
# - LR: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
# 	- für imbalanced dataset
#	- vllt als Lösung für weitere Verbesserungen?
# - DL Modelle hinzufügen! oder doch nciht?
# - mehr hyperparameters zum tunen eingeben

#!/usr/bin/env python
import argparse
from collections import defaultdict
from datetime import datetime
import logging
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score 
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

import sys
import time
# TODO
#from visualization import visualize


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
	# TODO überprüfen
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
					   "clf__C": [1.0],
					   "clf__max_iter": [1000]}

	lsvm_grid = GridSearchCV(lsvm_pipe, 
							 lsvm_parameters,
							 cv=cv, 
							 scoring="f1_macro")

	lsvm_cross_val1 = np.mean(cross_val_score(lsvm_grid, 
											  features, 
											  class1, 
											  cv=cv, 
											  scoring="f1_macro"))

	lsvm_cross_val2 = np.mean(cross_val_score(lsvm_grid, 
											  features, 
											  class2, 
											  cv=cv, 
											  scoring="f1_macro"))


	cv_dict["LSVM"] = {"year": lsvm_cross_val1,
					   "poet": lsvm_cross_val2}


	lsvm_duration = float(time.time() - lsvm_st)
	clf_durations["LSVM"].append(lsvm_duration)
	logging.info(f"Run-time LSVM: {lsvm_duration} seconds")

	# =====================
	# Logistic Regression #
	# =====================


	lr_st = time.time()

	lr_pipe = Pipeline(steps=[("vect", vectorizer),
						   	  ("clf", LogisticRegression())])


	lr_parameters = {"clf__penalty": ["l2"],
				     "clf__max_iter": [1000]}

	lr_grid = GridSearchCV(lr_pipe, 
						   lr_parameters,
						   cv=cv, 
						   scoring="f1_macro")

	lr_cross_val1 = np.mean(cross_val_score(lr_grid, 
										 	features, 
										 	class1, 
										 	cv=cv, 
										 	scoring="f1_macro"))

	lr_cross_val2 = np.mean(cross_val_score(lr_grid, 
											features, 
											class2, 
											cv=cv, 
											scoring="f1_macro"))

	cv_dict["LR"] = {"year": lr_cross_val1,
					 "poet": lr_cross_val2}

	lr_duration = float(time.time() - lr_st)
	clf_durations["LR"].append(lr_duration)
	logging.info(f"Run-time LR: {lr_duration} seconds")


	#TODO Visualisieren


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
	parser.add_argument("--n_jobs", "-nj", type=int, default=1, help="Indicates the number of processors used for computation.")
	parser.add_argument("--save_date", "-sd", action="store_true", help="Indicates if the creation date of the results should be saved.")
	parser.add_argument("--visualization", "-v", action="store_true", help="Indicates if results should be visualized.")
	
	args = parser.parse_args()

	main()