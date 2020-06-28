# bertpoems

Term paper for the Course "Textklassifikation" of the Julius-Maximilians-University, WS 19/20.

## project structure

- **app**.  This directory contains the scripts `bertclf.py`, `bert_opt.py` and `mlclf.py` used for classification and optimization as well as the helper script `utils.py`.  The notebooks were used for the preprocessing of the corpora, the creation of the graphics and the evaluation.
- **corpora**.  This folder contains the corpora used in the term paper.  The subfolder `domain-adaption` contains files used for domain adaptation and the PyTorch models `german` and `speech` adapted to the domain of the poetry corpus. The other folders contain the splitted corpora used for cross-validation.
- **resources**. This folder contains the graphics used in this term paper.
- **results**. This folder contains the results of the classification experiments as `json` and `csv` files.

	
