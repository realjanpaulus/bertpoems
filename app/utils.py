import datetime
import glob
import io
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.metrics.pairwise import cosine_similarity


# clf utils
def early_stopping(d, patience=2):
    """Implements Early stopping."""
    if len(d) <= 1:
        return False
    elif len(d) > 1:
        comparisons = []
        for epoch in range(1, len(d) + 1):
            if epoch > 1:
                comparisons.append(d[f"epoch{epoch}"] >= d[f"epoch{epoch-1}"])
        if False not in comparisons[-patience:] and len(comparisons) > patience:
            return True
        else:
            return False


def flat_f1(true_labels, preds):
    """Flattens predictions and labels and omputes macro f1-score."""
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = true_labels.flatten()
    return f1_score(labels_flat, pred_flat, average="macro")


def format_time(elapsed):
    """Takes a time in seconds and returns a string hh:mm:ss"""
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))


def load_train(path, cv, i, string):
    """Load a bunch of csv data files and add them together in a DataFrame.
    For optimal use, 'cv' should be greater than 1.
    """
    if cv == 1:
        print("ATTENTION! Train set is equal to val set.")
        return pd.read_csv(f"{path}/{string}{i}.csv")
    else:
        dfs = list()
        for f in range(1, cv + 1):
            if f != i:
                df = pd.read_csv(f"{path}/{string}{f}.csv")
                dfs.append(df)
        return pd.concat(dfs, axis=0, ignore_index=True)


# others


def df_to_latex(df, alignment="c"):
    """Convert a pandas dataframe to a LaTeX tabular.

    Notes
    -----
    Prints labels in bold, does not use math mode.
    Adapted from: https://techoverflow.net/2013/12/08/converting-a-pandas-dataframe-to-a-customized-latex-tabular/.
    """

    numColumns = df.shape[1]
    numRows = df.shape[0]
    output = io.StringIO()
    colFormat = "%s|%s" % (alignment, alignment * numColumns)
    # Write header
    output.write("\\small\n")
    output.write("\\begin{tabular}{%s}\n" % colFormat)
    output.write("\\hline\n")
    columnLabels = ["\\textbf{%s}" % label for label in df.columns]
    output.write("& %s\\\\\\hline\n" % " & ".join(columnLabels))
    # Write data lines
    for i in range(numRows):
        output.write(
            "\\textbf{%s} & %s\\\\\n" % (df.index[i], " & ".join([str(val) for val in df.iloc[i]]))
        )

    # Write footer
    output.write("\\end{tabular}")
    return output.getvalue()
