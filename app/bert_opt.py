import argparse
from itertools import product
import logging
import subprocess
import sys
import time

def main():

	program_st = time.time()


	logging.basicConfig(level=logging.DEBUG, 
						filename=f"../logs/run_{args.corpus_name}.log", 
						filemode="w")
	console = logging.StreamHandler()
	console.setLevel(logging.INFO)
	formatter = logging.Formatter("%(levelname)s: %(message)s")
	console.setFormatter(formatter)
	logging.getLogger('').addHandler(console)
	

	logging.info("Starting optimization.")
	batch_sizes = [4,8]
	learning_rates = [4e-5, 3e-5, 2e-5]

	cartesian_inputs = list(product(batch_sizes, learning_rates))

	
	for idx, t in enumerate(cartesian_inputs):
		
		print("--------------------------------------------")
		logging.info(f"Argument combination {idx+1}/{len(cartesian_inputs)}.")
		logging.info(f"Batch size: {t[0]}.")
		logging.info(f"Learning rate: {t[1]}.")
		print("--------------------------------------------")

		if args.domain_adaption:
			if args.domain_adaption_alternative_path:
				command = f"python bertclf.py -cn {args.corpus_name} -m {args.model} -da -daap -bs {t[0]} -lr {t[1]}"
			else:
				command = f"python bertclf.py -cn {args.corpus_name} -m {args.model} -da -bs {t[0]} -lr {t[1]}"
		else:
			command = f"python bertclf.py -cn {args.corpus_name} -m {args.model} -bs {t[0]} -lr {t[1]}"
		
		
		if args.save_date:
			command += " -sd"

		if args.save_confusion_matrices:
			command += " -scm"

		if args.save_misclassification:
			command += " -sm"

		#TODO
		command += " -e 1 -cv 2"
		
		subprocess.call(["bash", "-c", command])
		print("\n")
	program_duration = float(time.time() - program_st)
	logging.info(f"Overall run-time: {int(program_duration)/60} minute(s).")

	
	
if __name__ == "__main__":
	
	parser = argparse.ArgumentParser(prog="bert_opt", description="Runs bert classification script with multiple arguments.")
	parser.add_argument("--corpus_name", "-cn", type=str, default="poet", help="Indicates the corpus. Default is 'poet'. Another possible value is 'year'.")
	parser.add_argument("--domain_adaption", "-da", action="store_true", help="Indicates if a domain-adapted model should be used. '--domain_adapted_path' must be specified.")
	parser.add_argument("--domain_adaption_alternative_path", "-daap", action="store_true", help="Uses an alternative path if an pytorch model loading error occurs (e.g. git lfs is not installed).")
	parser.add_argument("--model", "-m", type=str, default="rede", help="Indicates the BERT model name. Default is 'rede' (short for: bert-base-historical-german-rw-cased). Another option is 'deutsch' (short for: bert-base-german-dbmdz-cased).")
	parser.add_argument("--save_confusion_matrices", "-scm", action="store_true", help="Indicates if confusion matrices should be saved." )
	parser.add_argument("--save_date", "-sd", action="store_true", help="Indicates if the creation date of the results should be saved.")
	parser.add_argument("--save_misclassification", "-sm", action="store_true", help="Indicates if pids of missclassifications should be saved.")

	args = parser.parse_args()

	main()