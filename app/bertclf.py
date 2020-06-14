#!/usr/bin/env python
# Some parts of the code are from this tutorial: 
# https://colab.research.google.com/drive/1pTuQhug6Dhl9XalKB0zUGf4FIdYFlpcX#scrollTo=6O_NbXFGMukX
import argparse
from collections import Counter, defaultdict
from datetime import datetime
import json
import logging
import numpy as np
import pandas as pd

from sklearn.metrics import f1_score 
from sklearn.model_selection import cross_val_score, cross_validate, StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder

#import tensorflow as tf
import torch
from torch.utils.data import TensorDataset, random_split, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, BertConfig, get_linear_schedule_with_warmup

import sys
import random
import time
import utils

def main():

	# ================
	# time managment #
	# ================

	program_st = time.time()

	# =====================================
	# bert classification logging handler #
	# =====================================
	logging_filename = f"../logs/bertclf_{args.corpus_name}.log"
	logging.basicConfig(level=logging.DEBUG, filename=logging_filename, filemode="w")
	console = logging.StreamHandler()
	console.setLevel(logging.INFO)
	formatter = logging.Formatter("%(levelname)s: %(message)s")
	console.setFormatter(formatter)
	logging.getLogger('').addHandler(console)

	mpl_logger = logging.getLogger('matplotlib')
	mpl_logger.setLevel(logging.WARNING)

	# =======================
	# predefined parameters #
	# =======================

	cv = 10
	num_labels = 3

	batch_size = args.batch_size
	epochs = args.epochs
	max_length = args.max_length
	

	if args.domain_adaption:
		if args.model == "german":
			#TODO: überprüfen
			model_name = '../corpora/domain-adaption/german/'
		elif args.model == "rede":
			#TODO: überprüfen
			model_name = '../corpora/domain-adaption/redewiedergabe/'
		else:
			logging.warning(f"Couldn't find a model with the name '{args.model}'.")
	else:
		if args.model == "german":
			model_name = 'bert-base-german-dbmdz-cased'
		elif args.model == "rede":
			model_name = 'redewiedergabe/bert-base-historical-german-rw-cased'
		else:
			logging.warning(f"Couldn't find a model with the name '{args.model}'.")
	

	

	cv_acc_dict = defaultdict(list)
	year_cv_dict = {}
	poet_cv_dict = {}

	# ================
	# classification # 
	# ================

	for i in range(1, cv+1):

		if args.corpus_name == "poet":
			train_data = utils.load_train("../corpora/train_epochpoet", cv, i, "epochpoet")
			val_data = pd.read_csv(f"../corpora/train_epochpoet/epochpoet{i}.csv")
		elif args.corpus_name == "year":
			train_data = utils.load_train("../corpora/train_epochyear", cv, i, "epochyear")
			val_data = pd.read_csv(f"../corpora/train_epochyear/epochyear{i}.csv")
		else:
			logging.warning(f"Couldn't find a corpus with the name '{args.corpus_name}'.")


		class_name1 = "epoch_year"
		class_name2 = "epoch_poet"
		text_name = "poem"

		# =======================
		# use GPU, if available #
		# =======================

		if torch.cuda.is_available(): 
			device = torch.device("cuda")  
			print('There are %d GPU(s) available.' % torch.cuda.device_count())
			print('Used GPU:', torch.cuda.get_device_name(0))
		else:
			print('No GPU available, using the CPU instead.')
			device = torch.device("cpu")


		for class_name in [class_name1, class_name2]:

			# tmp lists and result dicts #
			train_input_ids = []
			train_attention_masks = []
			val_input_ids = []
			val_attention_masks = []

			X_train = train_data[text_name].values
			X_val = val_data[text_name].values


			y_train = LabelEncoder().fit_transform(train_data[class_name].values)
			y_val = LabelEncoder().fit_transform(val_data[class_name].values)


			# ==============
			# tokenization #
			# ==============

			tokenizer = BertTokenizer.from_pretrained(model_name, do_lower_case=False)

			for sent in X_train:
				train_encoded = tokenizer.encode_plus(sent,
													  add_special_tokens = True,
													  max_length = args.max_length,
													  pad_to_max_length = True,
													  return_attention_mask = True, 
													  return_tensors = 'pt')

				train_input_ids.append(train_encoded['input_ids'])
				train_attention_masks.append(train_encoded['attention_mask'])

			# data leakage, da von gleichem tokenizer?
			for sent in X_val:
				val_encoded = tokenizer.encode_plus(sent,
													add_special_tokens = True,
													max_length = args.max_length,
													pad_to_max_length = True,
													return_attention_mask = True, 
													return_tensors = 'pt')

				val_input_ids.append(val_encoded['input_ids'])
				val_attention_masks.append(val_encoded['attention_mask'])

			# ==============================
			# filling tensors & DataLoader #
			# ==============================

			train_input_ids = torch.cat(train_input_ids, dim=0)
			val_input_ids = torch.cat(val_input_ids, dim=0)

			train_attention_masks = torch.cat(train_attention_masks, dim=0)
			val_attention_masks = torch.cat(val_attention_masks, dim=0)

			y_train = torch.tensor(y_train)
			y_val = torch.tensor(y_val)


			train = TensorDataset(train_input_ids, 
								  train_attention_masks,
								  y_train)

			val = TensorDataset(val_input_ids, 
								val_attention_masks,
								y_val)


			train_dataloader = DataLoader(train,
										  sampler = RandomSampler(train),
										  batch_size = batch_size)

			val_dataloader = DataLoader(val, 
										sampler = SequentialSampler(val),
										batch_size = batch_size)

			# ======== #
			# Training #
			# ======== #
			model = BertForSequenceClassification.from_pretrained(model_name, 
																  num_labels = num_labels,
																  output_attentions = False,
																  output_hidden_states = False).cuda()

			optimizer = AdamW(model.parameters(),
							  lr = 2e-5,
							  eps = 1e-8)

			total_steps = len(train_dataloader) * epochs

			scheduler = get_linear_schedule_with_warmup(optimizer, 
														num_warmup_steps = 0, 
														num_training_steps = total_steps)

			
			training_stats = []
			total_t0 = time.time()

			for epoch_i in range(0, epochs):
				print("")
				print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
				print('Now Training.')
				t0 = time.time()
				total_train_loss = 0
				model.train()
				for step, batch in enumerate(train_dataloader):
					if step % 20 == 0 and not step == 0:
						elapsed = utils.format_time(time.time() - t0)
						print('Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

					b_input_ids = batch[0].to(device)
					b_input_mask = batch[1].to(device)
					b_labels = batch[2].to(device)

					model.zero_grad()        

					loss, logits = model(b_input_ids, 
										 token_type_ids=None, 
										 attention_mask=b_input_mask, 
										 labels=b_labels)

					total_train_loss += loss.item()
					loss.backward()
					torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
					optimizer.step()
					scheduler.step()

				# average loss (all batches)
				avg_train_loss = total_train_loss / len(train_dataloader)   
				training_time = utils.format_time(time.time() - t0)

				print("")
				print("  Average training loss: {0:.2f}".format(avg_train_loss))
				print("  Training epoch took: {:}".format(training_time))
				
				# ========== #
				# Validation #
				# ========== #

				print("")
				print("Now Validating.")

				t0 = time.time()
				model.eval()

				total_eval_accuracy = 0
				total_eval_loss = 0
				nb_eval_steps = 0

				for batch in val_dataloader:
					
					b_input_ids = batch[0].to(device)
					b_input_mask = batch[1].to(device)
					b_labels = batch[2].to(device)
					
					with torch.no_grad():        

						(loss, logits) = model(b_input_ids, 
											   token_type_ids=None, 
											   attention_mask=b_input_mask,
											   labels=b_labels)
						
					# validation loss.
					total_eval_loss += loss.item()

					# Move logits and labels to CPU
					logits = logits.detach().cpu().numpy()
					label_ids = b_labels.to('cpu').numpy()

					# TODO: geht f1 score hier?
					total_eval_accuracy += utils.flat_f1(logits, label_ids)
					

				# final validation accuracy / loss
				avg_val_accuracy = total_eval_accuracy / len(val_dataloader)
				print("  Validation Accuracy: {0:.2f}".format(avg_val_accuracy))
				
				avg_val_loss = total_eval_loss / len(val_dataloader)
				validation_time = utils.format_time(time.time() - t0)
				print("  Validation Loss: {0:.2f}".format(avg_val_loss))
				print("  Validation took: {:}".format(validation_time))

				
				training_stats.append({'epoch': epoch_i + 1,
									   'train_loss': avg_train_loss,
									   'val_loss': avg_val_loss,
									   'val_acc': avg_val_accuracy,
									   'train_time': training_time,
									   'val_time': validation_time})

			logging.info(f"Training for {class_name} done.")
			logging.info("Training took {:} (h:mm:ss)".format(utils.format_time(time.time()-total_t0)))
			
			stats = pd.DataFrame(data=training_stats)
			cv_acc_dict[class_name].append(utils.get_mean_acc(stats))

			if class_name == "epoch_year":
				year_cv_dict[f"cv{i}"] = training_stats
			elif class_name == "epoch_poet":
				poet_cv_dict[f"cv{i}"] = training_stats
			else:
				logging.info(f"The class {class_name} does not exist.")

		
		logging.info(f"Training for run {i}/{cv} completed.")
		logging.info("Training run took {:} (h:mm:ss)".format(utils.format_time(time.time()-total_t0)))
	

	# ================
	# saving results #
	# ================

	result_path = "../results/bert/"
	if args.domain_adaption:
		output_name = f"{args.corpus_name}_da_{args.model}"
	else:
		output_name = f"{args.corpus_name}_{args.model}"

	if args.save_date:
		output_name += f"({datetime.now():%d.%m.%y}_{datetime.now():%H:%M})"


	with open(f'{result_path}cv_{output_name}.json', 'w') as f:
		json.dump(cv_acc_dict, f)

	with open(f'{result_path}year_{output_name}.json', 'w') as f:
		json.dump(year_cv_dict, f)

	with open(f'{result_path}poet{output_name}.json', 'w') as f:
		json.dump(poet_cv_dict, f)

	program_duration = float(time.time() - program_st)
	logging.info(f"Total duration: {int(program_duration)/60} minute(s).")



if __name__ == "__main__":
	
	parser = argparse.ArgumentParser(prog="bertclf", description="Bert classifier.")
	parser.add_argument("--batch_size", "-bs", type=int, default=8, help="Indicates batch size.")
	parser.add_argument("--corpus_name", "-cn", type=str, default="year", help="Indicates the corpus. Default is 'year'. Another possible value is 'poet'.")
	parser.add_argument("--domain_adaption", "-da", action="store_true", help="Indicates if a domain-adapted model should be used. '--domain_adapted_path' must be specified.")
	parser.add_argument("--domain_adapted_path", "-dap", type=str, default="../corpora/domain-adaption", help="Indicates the path of a domain-adapted model.")
	parser.add_argument("--epochs", "-e", type=int, default=4, help="Indicates number of epochs.")
	parser.add_argument("--max_length", "-ml", type=int, default=510, help="Indicates the maximum document length.")
	parser.add_argument("--model", "-m", type=str, default="german", help="Indicates the BERT model name. Default is 'german' (short for: bert-base-german-dbmdz-cased). Another option is 'rede' (short for: bert-base-historical-german-rw-cased).")
	parser.add_argument("--save_date", "-sd", action="store_true", help="Indicates if the creation date of the results should be saved.")
	
	args = parser.parse_args()

	main()