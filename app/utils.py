"""TODO
- alles überprüfen
"""

from collections import defaultdict
from datetime import datetime
import glob
import io
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from typing import Dict, List, Optional, Tuple, Union


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