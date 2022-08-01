# bertpoems

In a series of experiments, it was investigated whether it is possible to identify **literary epochs** of **poems** using **text classification techniques**. The following techniques were used:
- BERT
- SVM
- Logistic Regression

In addition, experiments with **BERT** were extended with a **domain adaption**. The detailed analysis of the text classification techniques as well as a comparison of these methods were carried out in a seminar paper (access only on request).

## Project Structure

- **`app`**. Contains the scripts `bertclf.py`, `bert_opt.py` and `mlclf.py` used for classification and optimization as well as the helper script `utils.py`.  The notebooks were used for the preprocessing of the corpora, the creation of the figures and the evaluation.
- **`corpora`**. Contains the corpora used in the term paper.  The subfolder `domain-adaption` contains files used for domain adaptation and the PyTorch models `german` and `speech` adapted to the domain of the poetry corpus. The other folders contain the splitted corpora used for cross-validation.
- **`resources`**. Contains the figures used in this term paper.
- **`results`**. Contains the results of the classification experiments as `json` and `csv` files.

## Virtual Environment Setup

Create and activate the environment (the python version and the environment name can vary at will):

```sh
$ python3.9 -m venv .env
$ source .env/bin/activate
```

To install the project's dependencies, activate the virtual environment and simply run (requires [poetry](https://python-poetry.org/)):

```sh
$ poetry install
```

Alternatively, use the following:

```sh
$ pip install -r requirements.txt
```

Deactivate the environment:

```sh
$ deactivate
```


## Notes

`scikit-learn` isn't part of the `pyproject.toml` due to the support of Apple's M1 chip and has to be installed manually:

```sh
pip install scikit-learn
```
