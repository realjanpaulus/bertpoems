from collections import defaultdict
import datetime
import glob
import io
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from sklearn.metrics import f1_score 
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import time
from typing import Dict, List, Optional, Tuple, Union

# ===========
# clf utils #
# ===========

def flat_f1(preds, labels):
    """ Flattens predictions and labels and omputes macro f1-score.
    """
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return f1_score(pred_flat, labels_flat, average="macro")


def format_time(elapsed):
    """ Takes a time in seconds and returns a string hh:mm:ss
    """
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))

def get_mean_acc(df):
    """ Returns mean validation accuracy for non-overfitted training.
    """
    prev_loss = 1000
    val_acc = []
    for idx, row in df.iterrows():
        val_acc.append(row["val_acc"])
        if row["val_loss"] > prev_loss:
            break
        else:
            prev_loss = row["val_loss"]
    return np.mean(val_acc)


def load_train(path, cv, i, string):
    """ Load a bunch of csv data files and add them together in a DataFrame.
        For optimal use, 'cv' should be greater than 1.
    """
    if cv == 1:
        print("ATTENTION! Train set is equal to val set.")
        return pd.read_csv(f"{path}/{string}{i}.csv")
    else:
        dfs = list()
        for f in range(1, cv+1):
            if f != i:
                df = pd.read_csv(f"{path}/{string}{f}.csv")   
                dfs.append(df)
        return pd.concat(dfs, axis=0, ignore_index=True)

# ========
# others #
# ========

def df_to_latex(df, alignment="c"):
    """ Convert a pandas dataframe to a LaTeX tabular.
        Prints labels in bold, does not use math mode.
        Adapted from: https://techoverflow.net/2013/12/08/converting-a-pandas-dataframe-to-a-customized-latex-tabular/.
    """

    numColumns = df.shape[1]
    numRows = df.shape[0]
    output = io.StringIO()
    colFormat = ("%s|%s" % (alignment, alignment * numColumns))
    #Write header
    output.write("\\small\n")
    output.write("\\begin{tabular}{%s}\n" % colFormat)
    output.write("\\hline\n")
    columnLabels = ["\\textbf{%s}" % label for label in df.columns]
    output.write("& %s\\\\\\hline\n" % " & ".join(columnLabels))
    #Write data lines
    for i in range(numRows):
        output.write("\\textbf{%s} & %s\\\\\n"
                     % (df.index[i], " & ".join([str(val) for val in df.iloc[i]])))

    #Write footer
    output.write("\\end{tabular}")
    return output.getvalue()