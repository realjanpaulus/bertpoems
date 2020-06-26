#!/usr/bin/env python
# Some parts of the code are from this tutorial: 
# https://colab.research.google.com/drive/1pTuQhug6Dhl9XalKB0zUGf4FIdYFlpcX#scrollTo=6O_NbXFGMukX
import argparse
from collections import Counter, defaultdict
from datetime import datetime
from keras.preprocessing.sequence import pad_sequences
import json
import logging
import numpy as np
import pandas as pd

from sklearn.metrics import confusion_matrix, f1_score 
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
	logging.basicConfig(level=logging.INFO, filename=logging_filename, filemode="w")
	console = logging.StreamHandler()
	console.setLevel(logging.INFO)
	formatter = logging.Formatter("%(levelname)s: %(message)s")
	console.setFormatter(formatter)
	logging.getLogger('').addHandler(console)


	# =======================
	# predefined parameters #
	# =======================

	cv = args.cross_validation
	num_labels = 3

	batch_size = args.batch_size
	epochs = args.epochs
	learning_rate = args.learning_rate
	max_length = args.max_length

	

	if args.domain_adaption:
		if args.model == "german":
			if args.domain_adaption_alternative_path:
				model_name = '../corpora/domain-adaption/german-alternative/'
			else:
				model_name = '../corpora/domain-adaption/german/'
		elif args.model == "rede":
			if args.domain_adaption_alternative_path:
				model_name = '../corpora/domain-adaption/redewiedergabe-alternative/'
			else:
				model_name = '../corpora/domain-adaption/redewiedergabe/'
		elif args.model == "test":
			model_name = '../corpora/domain-adaption/test/'
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

	for i in range(1, cv+1):

		if args.corpus_name == "poet":
			train_data = utils.load_train("../corpora/train_epochpoet", cv, i, "epochpoet")
			test_data = pd.read_csv(f"../corpora/train_epochpoet/epochpoet{i}.csv")
		elif args.corpus_name == "year":
			train_data = utils.load_train("../corpora/train_epochyear", cv, i, "epochyear")
			test_data = pd.read_csv(f"../corpora/train_epochyear/epochyear{i}.csv")
		else:
			logging.warning(f"Couldn't find a corpus with the name '{args.corpus_name}'.")


		class_name1 = "epoch_year"
		class_name2 = "epoch_poet"
		text_name = "poem"

		#TODO
		for class_name in [class_name2]:#[class_name1, class_name2]:

			# tmp lists and result dicts #
			input_ids = []
			attention_masks = []

			texts = train_data[text_name].values
			labels = LabelEncoder().fit_transform(train_data[class_name].values)

			# ==============
			# tokenization #
			# ==============

			tokenizer = BertTokenizer.from_pretrained(model_name, do_lower_case=False)

			for sent in texts:
				encoded_dict = tokenizer.encode_plus(sent,
													 add_special_tokens = True,
													 max_length = args.max_length,
													 pad_to_max_length = True,
													 return_attention_mask = True, 
													 return_tensors = 'pt')

				input_ids.append(encoded_dict['input_ids'])
				attention_masks.append(encoded_dict['attention_mask'])

	
			input_ids = torch.cat(input_ids, dim=0)
			attention_masks = torch.cat(attention_masks, dim=0)
			labels = torch.tensor(labels)

			# =================
			# train val split #
			# =================

			dataset = TensorDataset(input_ids, attention_masks, labels)

			train_size = int(0.9 * len(dataset))
			val_size = len(dataset) - train_size

			train_dataset, val_dataset = random_split(dataset, [train_size, val_size])


			# ============
			# DataLoader #
			# ============

			train_dataloader = DataLoader(train_dataset,
										  sampler = RandomSampler(train_dataset),
										  batch_size = batch_size)

			val_dataloader = DataLoader(val_dataset, 
										sampler = SequentialSampler(val_dataset),
										batch_size = batch_size)

			# ======== #
			# Training #
			# ======== #
			model = BertForSequenceClassification.from_pretrained(model_name, 
																  num_labels = num_labels,
																  output_attentions = False,
																  output_hidden_states = False).cuda()

			optimizer = AdamW(model.parameters(),
							  lr=learning_rate,
							  eps=1e-8)

			total_steps = len(train_dataloader) * epochs

			scheduler = get_linear_schedule_with_warmup(optimizer, 
														num_warmup_steps = 0, 
														num_training_steps = total_steps)

			
			training_stats = []
			total_t0 = time.time()

			validation_losses = {}

			for epoch_i in range(0, epochs):
				print("")
				print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
				print('Now Training.')
				t0 = time.time()
				total_train_loss = 0
				model.train()
				for step, batch in enumerate(train_dataloader):
					if step % 50 == 0 and not step == 0:
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

					total_eval_accuracy += utils.flat_f1(label_ids, logits)
					

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

				current_epoch = f"epoch{epoch_i + 1}"
				validation_losses[current_epoch] = avg_val_loss

				# ================
				# Early Stopping #
				# ================


				if utils.early_stopping(validation_losses, patience=2):
					logging.info(f"Stopping epoch run early (Epoch {epoch_i}).")
					break


			logging.info(f"Training for {class_name} done.")
			logging.info("Training took {:} (h:mm:ss) \n".format(utils.format_time(time.time()-total_t0)))
			print("--------------------------------\n")

			# =========
			# Testing #
			# =========

			test_input_ids = []
			test_attention_masks = []

			X_test = test_data[text_name].values
			y_test = LabelEncoder().fit_transform(test_data[class_name].values)


			for sent in X_test:
				encoded_dict = tokenizer.encode_plus(sent,
													 add_special_tokens = True,
													 max_length = args.max_length,
													 pad_to_max_length = True,
													 return_attention_mask = True, 
													 return_tensors = 'pt')
	
				test_input_ids.append(encoded_dict['input_ids'])
				
				test_attention_masks.append(encoded_dict['attention_mask'])

			test_input_ids = torch.cat(test_input_ids, dim=0)
			test_attention_masks = torch.cat(test_attention_masks, dim=0)
			labels = torch.tensor(y_test)

			prediction_data = TensorDataset(test_input_ids, test_attention_masks, labels)
			prediction_sampler = SequentialSampler(prediction_data)
			prediction_dataloader = DataLoader(prediction_data, 
											   sampler=prediction_sampler, 
											   batch_size=batch_size)

			model.eval()

			predictions, true_labels = [], []

			for batch in prediction_dataloader:
				# Add batch to GPU
				batch = tuple(t.to(device) for t in batch)
				  
				# Unpack the inputs from our dataloader
				b_input_ids, b_input_mask, b_labels = batch
				  
				
				with torch.no_grad():
					outputs = model(b_input_ids, 
									token_type_ids=None, 
									attention_mask=b_input_mask)

				logits = outputs[0]

				# Move logits and labels to CPU
				logits = logits.detach().cpu().numpy()
				label_ids = b_labels.to('cpu').numpy()
				  
				# Store predictions and true labels
				predictions.append(logits)
				true_labels.append(label_ids)


			test_pid = test_data["pid"].values
			print(test_pid)
			"""
			for j in range(len(true_labels)):
				pred_labels_j = np.argmax(predictions[j], axis=1).flatten()
			"""
			print(X_test.shape)
			print("hier----------------------")
			print([np.argmax(predictions[j], axis=1).flatten() for j in range(len(true_labels))])
			print("\n")
			print(true_labels)
			print("---------------------------")
			pr = [i for j in range(len(true_labels)) for i in np.argmax(predictions[j], axis=1).flatten()]
			tr = [i for l in true_labels for i in l]
			print(f1_score(tr, pr, average="macro"))

			scores = []
			cmatrices = []

			# looping through predictions #
			for j in range(len(true_labels)):
				pred_labels_j = np.argmax(predictions[j], axis=1).flatten()
				f1_j = f1_score(true_labels[j], pred_labels_j, average="macro")             
				scores.append(f1_j)

				cm = confusion_matrix(true_labels[j], pred_labels_j)
				if cm.shape == (2,2):
					cm = np.lib.pad(cm, ((0,1),(0,1)), 'constant', constant_values=(0))
				elif cm.shape == (1,1):
					cm = np.lib.pad(cm, ((0,2),(0,2)), 'constant', constant_values=(0))
				cmatrices.append(cm)

			
			test_score = np.mean(scores)			
			classes = test_data[class_name].drop_duplicates().tolist()
			cm_sum = sum(cmatrices)
			cm_df = pd.DataFrame(cm_sum, index=classes, columns=classes)

			if args.domain_adaption:
				cm_name = f"{args.corpus_name}c_{class_name}_da_{args.model}"
			else:
				cm_name = f"{args.corpus_name}c_{class_name}_{args.model}"

			if args.save_date:
				cm_name += f"({datetime.now():%d.%m.%y}_{datetime.now():%H:%M})"

			cm_df.to_csv(f"../results/bert/confusion_matrices/cm{i}_{cm_name}.csv")
			

			stats = pd.DataFrame(data=training_stats)
			cv_acc_dict[class_name].append(test_score)


			if class_name == "epoch_year":
				year_cv_dict[f"cv{i}"] = training_stats
			elif class_name == "epoch_poet":
				poet_cv_dict[f"cv{i}"] = training_stats
			else:
				logging.info(f"The class {class_name} does not exist.")

			logging.info(f"Testing for {class_name} done.")
			logging.info(f"CV Test F1-Score: {test_score} (run: {i}/{cv}).")
			logging.info("Testing took {:} (h:mm:ss) \n".format(utils.format_time(time.time()-total_t0)))
			print("--------------------------------\n")

		logging.info(f"Training for run {i}/{cv} completed.")
		logging.info("Training run took {:} (h:mm:ss)".format(utils.format_time(time.time()-total_t0)))
		print("________________________________")
		print("________________________________\n")

	# ================
	# saving results #
	# ================

	result_path = "../results/bert/"
	logging.info(f"Writing results to '{result_path}'.")
	if args.domain_adaption:
		output_name = f"{args.corpus_name}c_da_{args.model}"
	else:
		output_name = f"{args.corpus_name}c_{args.model}"

	if args.save_date:
		output_name += f"({datetime.now():%d.%m.%y}_{datetime.now():%H:%M})"


	with open(f'{result_path}cv_{output_name}.json', 'w') as f:
		json.dump(cv_acc_dict, f)

	with open(f'{result_path}eyear_{output_name}.json', 'w') as f:
		json.dump(year_cv_dict, f)

	with open(f'{result_path}epoet_{output_name}.json', 'w') as f:
		json.dump(poet_cv_dict, f)

	program_duration = float(time.time() - program_st)
	logging.info(f"Total duration: {int(program_duration)/60} minute(s).")



if __name__ == "__main__":
	
	parser = argparse.ArgumentParser(prog="bertclf", description="Bert classifier.")
	parser.add_argument("--batch_size", "-bs", type=int, default=8, help="Indicates batch size.")
	parser.add_argument("--corpus_name", "-cn", type=str, default="year", help="Indicates the corpus. Default is 'year'. Another possible value is 'poet'.")
	parser.add_argument("--cross_validation", "-cv", type=int, default=10, help="Indicates the number of cross validations.")
	parser.add_argument("--domain_adaption", "-da", action="store_true", help="Indicates if a domain-adapted model should be used. '--domain_adapted_path' must be specified.")
	parser.add_argument("--domain_adaption_alternative_path", "-daap", action="store_true", help="Uses an alternative path if an pytorch model loading error occurs (e.g. git lfs is not installed).")
	parser.add_argument("--epochs", "-e", type=int, default=10, help="Indicates number of epochs.")
	parser.add_argument("--learning_rate", "-lr", type=float, default=2e-5, help="Set learning rate for optimizer.")
	parser.add_argument("--max_length", "-ml", type=int, default=510, help="Indicates the maximum document length.")
	parser.add_argument("--model", "-m", type=str, default="german", help="Indicates the BERT model name. Default is 'german' (short for: bert-base-german-dbmdz-cased). Another option is 'rede' (short for: bert-base-historical-german-rw-cased).")
	parser.add_argument("--patience", "-p", type=int, default=3, help="Indicates patience for early stopping.")
	parser.add_argument("--save_date", "-sd", action="store_true", help="Indicates if the creation date of the results should be saved.")
	
	args = parser.parse_args()

	main()