#!/usr/bin/env python
import argparse
from collections import Counter, defaultdict
from datetime import datetime
import json
import logging
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score 
from sklearn.model_selection import cross_val_predict, cross_val_score, cross_validate, GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC

import sys
import time


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
								 stop_words=None)

	
	# ================================
	# classification logging handler #
	# ================================
	logging_filename = f"../logs/mlclf_lsvm_{args.corpus_name}.log"
	logging.basicConfig(level=logging.DEBUG, filename=logging_filename, filemode="w")
	console = logging.StreamHandler()
	console.setLevel(logging.INFO)
	formatter = logging.Formatter("%(levelname)s: %(message)s")
	console.setFormatter(formatter)
	logging.getLogger('').addHandler(console)

	
	# =================
	# corpus loading  # 
	# =================

	if args.corpus_name == "poet":
		corpus = pd.read_csv("../corpora/german_modern_poems_epochpoet.csv")
	elif args.corpus_name == "year":
		corpus = pd.read_csv("../corpora/german_modern_poems_epochyear.csv")
	elif args.corpus_name == "poeta":
		corpus = pd.read_csv("../corpora/german_modern_poems_epochpoet_alternative.csv")
	else:
		logging.warning(f"Couldn't find a corpus with the name '{args.corpus_name}'.")
	
	class_name2 = "epoch_poet"
	text_name = "poem"

	features = corpus[text_name]
	class2 = corpus[class_name2]

	logging.info(f"Read {args.corpus_name} corpus ({int((time.time() - program_st)/60)} minute(s)).")
	
	# ================
	# classification # 
	# ================


	# ============
	# Linear SVM #
	# ============

	lsvm_st = time.time()
	lsvm_pipe = Pipeline(steps=[("vect", vectorizer),
								("clf", LinearSVC())])
	
	lsvm_parameters = {"vect__ngram_range": [(1,1), (1,2), (2,2)],
					   "clf__penalty": ["l2"],
					   "clf__loss": ["squared_hinge"],
					   "clf__tol": [1e-5, 1e-3],
					   "clf__C": list(range(1, 11, 2)),
					   "clf__max_iter": [1000, 3000, 5000],
					   "clf__class_weight": [None, "balanced"]}
	#ALTERNATIVE
	"""
	lsvm_parameters = {"clf__penalty": ["l2"],
					   "clf__loss": ["squared_hinge"],
					   "clf__tol": [1e-3],
					   "clf__C": [1.0],
					   "clf__max_iter": [1000]}
	"""

	

	lsvm_grid2 = GridSearchCV(lsvm_pipe, 
							  lsvm_parameters,
							  cv=cv, 
							  error_score=0.0,
							  n_jobs=args.n_jobs,
							  scoring="f1_macro")

	lsvm_grid2.fit(features, class2)
	

	lsvm_cv_scores2 = cross_validate(lsvm_grid2, 
									  features, 
									  class2, 
									  cv=cv, 
									  return_estimator=False,
									  scoring="f1_macro")


	lsvm_preds = cross_val_predict(lsvm_grid2, 
								   features, 
								   class2, 
								   cv=cv, 
								   n_jobs=args.n_jobs)
	
	

	test_pid = corpus["pid"].values
	false_classifications = {"Jahrhundertwende": {"Naturalismus": [], "Expressionismus": []},
							 "Naturalismus": {"Jahrhundertwende": [], "Expressionismus": []},
							 "Expressionismus": {"Naturalismus": [], "Jahrhundertwende": []}}

	for idx, (t, p) in enumerate(zip(class2, lsvm_preds)):
		if t != p:
			false_classifications[t][p].append(int(test_pid[idx]))

	with open(f'../results/ml/misclassifications/pid_lsvm.json', 'w+') as f:
		json.dump(false_classifications, f)


	#todo: weg
	class2_unique = class2.drop_duplicates().tolist()

	lclass2 = class2


	conf_mat2 = confusion_matrix(lclass2, lsvm_cv_scores2)
	cm_df2 = pd.DataFrame(conf_mat2, index=class2_unique, columns=class2_unique)

	
	if args.save_date:
		output_path2 = f"../results/lsvm_cm_epa_{args.corpus_name}({datetime.now():%d.%m.%y}_{datetime.now():%H:%M}).csv"
	else:
		output_path2 = f"../results/lsvm_cm_epa_{args.corpus_name}.csv"
	cm_df2.to_csv(output_path2)


	cv_dict["LSVM"] = {"poet": np.mean(lsvm_cv_scores2["test_score"])}

	lsvm_duration = float(time.time() - lsvm_st)
	clf_durations["LSVM"].append(lsvm_duration)
	logging.info(f"Run-time LSVM: {lsvm_duration} seconds")

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
	
	parser = argparse.ArgumentParser(prog="mlclf_lsvm", description="Classification of LSVM.")
	parser.add_argument("--corpus_name", "-cn", type=str, default="year", help="Indicates the corpus. Default is 'year'. Other possible values are 'poet' or 'poeta'.")
	parser.add_argument("--lowercase", "-l", type=bool, default=False, help="Indicates if words should be lowercased.")
	parser.add_argument("--max_features", "-mf", type=int, default=60000, help="Indicates the number of most frequent words.")
	parser.add_argument("--n_jobs", "-nj", type=int, default=1, help="Indicates the number of processors used for computation.")
	parser.add_argument("--save_date", "-sd", action="store_true", help="Indicates if the creation date of the results should be saved.")
	
	args = parser.parse_args()

	main()
