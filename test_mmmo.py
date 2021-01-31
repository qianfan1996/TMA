# -*-coding:utf-8-*-
import numpy as np
import random
import json
from math import ceil
from tqdm import trange
import argparse

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, f1_score

from load_data import load_mmmo
from models import MFN, TMAN1, MARN, TMAN2

def set_random_seed(seed=666):
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)


def models_train(X_train, y_train, X_valid, y_valid, X_test, y_test, configs, args):

	set_random_seed()

	# train data random shuffle
	p = np.random.permutation(X_train.shape[0])
	X_train = X_train[p]
	y_train = y_train[p]

	X_train = X_train.swapaxes(0, 1)
	X_valid = X_valid.swapaxes(0, 1)
	X_test = X_test.swapaxes(0, 1)

	if args.model == "mfn":
		model = MFN(*configs)
	elif args.model == "tman1":
		model = TMAN1(*configs)
	elif args.model == "marn":
		model = MARN(*configs)
	else:
		model = TMAN2(*configs)

	optimizer = optim.Adam(model.parameters(), lr=args.lr)

	criterion = nn.L1Loss()
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	model = model.to(device)
	criterion = criterion.to(device)

	def train(model, batchsize, X_train, y_train, optimizer, criterion):
		epoch_loss = 0
		model.train()
		total_n = X_train.shape[1]
		num_batches = ceil(total_n // batchsize)
		for batch in range(num_batches):
			start = batch*batchsize
			end = (batch+1)*batchsize
			optimizer.zero_grad()
			batch_X = torch.Tensor(X_train[:, start:end]).cuda()
			batch_y = torch.Tensor(y_train[start:end]).squeeze(-1).cuda()
			predictions = model.forward(batch_X).squeeze(1)
			loss = criterion(predictions, batch_y)
			loss.backward()
			optimizer.step()
			epoch_loss += loss.item()
		return epoch_loss / num_batches

	def validate(model, X_valid, y_valid, criterion):
		model.eval()
		with torch.no_grad():
			batch_X = torch.Tensor(X_valid).cuda()
			batch_y = torch.Tensor(y_valid).squeeze(-1).cuda()
			predictions = model.forward(batch_X).squeeze(1)
			epoch_loss = criterion(predictions, batch_y).item()
		return epoch_loss

	def predict(model, X_test):
		model.eval()
		with torch.no_grad():
			batch_X = torch.Tensor(X_test).cuda()
			predictions = model.forward(batch_X).squeeze(1)
			predictions = predictions.cpu().data.numpy()
		return predictions

	best_valid_loss = 999999.0
	for epoch in trange(args.epoch):
		train_loss = train(model, args.bs, X_train, y_train, optimizer, criterion)
		valid_loss = validate(model, X_valid, y_valid, criterion)
		if valid_loss <= best_valid_loss:
			best_valid_loss = valid_loss
			print(str(epoch)+'th epoch', ', train loss is', train_loss, ', valid loss is', valid_loss)
			print('Saving model...')
			torch.save(model, 'saved_models/' + args.model + '_mmmo.pt')
		else:
			print(str(epoch) + 'th epoch', ', train loss is', train_loss, ', valid loss is', valid_loss)

	model = torch.load('saved_models/' + args.model + '_mmmo.pt')

	predictions = predict(model, X_test)
	y_test = np.squeeze(y_test)
	mae = np.mean(np.absolute(predictions-y_test))
	print("MAE:", mae)
	corr = np.corrcoef(predictions,y_test)[0][1]
	print("corr:", corr)
	true_label = (y_test > 3.5)
	predicted_label = (predictions > 3.5)
	print("Confusion Matrix:")
	print(confusion_matrix(true_label, predicted_label))
	bi_acc = accuracy_score(true_label, predicted_label)
	print("Accuracy:", bi_acc)
	f1 = round(f1_score(true_label, predicted_label, average='binary'), 5)
	print("F1 score:", f1)


def models_test(X_test, y_test, args):
	X_test = X_test.swapaxes(0, 1)

	def predict(model, X_test):
		model.eval()
		with torch.no_grad():
			batch_X = torch.Tensor(X_test).cuda()
			predictions = model.forward(batch_X).squeeze(1)
			predictions = predictions.cpu().data.numpy()
		return predictions

	model = torch.load('saved_models/' + args.model + '_mmmo.pt')

	predictions = predict(model, X_test)
	y_test = np.squeeze(y_test)
	mae = np.mean(np.absolute(predictions - y_test))
	print("MAE:", mae)
	corr = np.corrcoef(predictions, y_test)[0][1]
	print("corr:", corr)
	true_label = (y_test > 3.5)
	predicted_label = (predictions > 3.5)
	print("Confusion Matrix:")
	print(confusion_matrix(true_label, predicted_label))
	print("Accuracy:", accuracy_score(true_label, predicted_label))
	f1 = round(f1_score(true_label, predicted_label, average='binary'), 5)
	print("F1 score:", f1)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("-m", "--model", choices=["mfn", "tman1", "marn", "tman2"], type=str, default="tman1", help="choose models")
	parser.add_argument("-t", "--train", action="store_false", help="train or evaluate models")
	parser.add_argument("--lr", type=int, default=0.001, help="learning_rate")
	parser.add_argument("--bs", type=int, default=32, help="batch_size")
	parser.add_argument("--epoch", type=int, default=50, help="num_epoch")

	args = parser.parse_args()

	X_train, y_train, X_valid, y_valid, X_test, y_test = load_mmmo()

	if args.train:
		if args.model == "mfn":
			Config = dict()
			Config["input_dims"] = [300, 74, 35]
			hl = 128
			ha = 64
			hv = 64
			Config["h_dims"] = [hl, ha, hv]
			Config["memsize"] = 100
			Config["shapes"] = 100
			Config["drop"] = 0.3
			NNpConfig = dict()
			NNpConfig["shapes"] = 50
			NNpConfig["drop"] = 0.7
			Gamma1Config = dict()
			Gamma1Config["shapes"] = 32
			Gamma1Config["drop"] = 0.3
			Gamma2Config = dict()
			Gamma2Config["shapes"] = 64
			Gamma2Config["drop"] = 0.7
			OutConfig = dict()
			OutConfig["shapes"] = 64
			OutConfig["drop"] = 0.3
			configs = [Config, NNpConfig, Gamma1Config, Gamma2Config, OutConfig]
			print(configs)

			models_train(X_train, y_train, X_valid, y_valid, X_test, y_test, configs, args)

		elif args.model == "tman1":
			Config = dict()
			Config["input_dims"] = [300, 74, 35]
			hl = 128
			ha = 64
			hv = 64
			Config["h_dims"] = [hl, ha, hv]
			Config["memsize"] = 100
			Config["sentisize"] = 200
			Config["shapes"] = 64
			Config["drop"] = 0.9
			NNlConfig = dict()
			NNlConfig["shapes"] = 200
			NNlConfig["drop"] = 0.0
			NNaConfig = dict()
			NNaConfig["shapes"] = 50
			NNaConfig["drop"] = 0.7
			NNvConfig = dict()
			NNvConfig["shapes"] = 64
			NNvConfig["drop"] = 0.5
			NNpConfig = dict()
			NNpConfig["shapes"] = 256
			NNpConfig["drop"] = 0.9
			Gamma1Config = dict()
			Gamma1Config["shapes"] = 32
			Gamma1Config["drop"] = 0.7
			Gamma2Config = dict()
			Gamma2Config["shapes"] = 200
			Gamma2Config["drop"] = 0.0
			OutConfig = dict()
			OutConfig["shapes"] = 64
			OutConfig["drop"] = 0.5
			configs = [Config, NNlConfig, NNaConfig, NNvConfig, NNpConfig, Gamma1Config, Gamma2Config, OutConfig]
			print(configs)

			models_train(X_train, y_train, X_valid, y_valid, X_test, y_test, configs, args)

		elif args.model == "marn":
			Config = dict()
			Config["input_dims"] = [300, 74, 35]
			hl = 128
			ha = 64
			hv = 64
			Config["h_dims"] = [hl, ha, hv]
			Config['num_atts'] = 4
			ReduceDimConfig = dict()
			r_hl = 32
			r_ha = 128
			r_hv = 32
			ReduceDimConfig["h_dims"] = [r_hl, r_ha, r_hv]
			MapNNConfig = dict()
			MapNNConfig["shapes"] = 256
			MapNNConfig["drop"] = 0.3
			OutConfig = dict()
			OutConfig["shapes"] = 64
			OutConfig["drop"] = 0.0
			configs = [Config, ReduceDimConfig, MapNNConfig, OutConfig]
			print(configs)

			models_train(X_train, y_train, X_valid, y_valid, X_test, y_test, configs, args)

		else:
			Config = dict()
			Config["input_dims"] = [300, 74, 35]
			hl = 128
			ha = 64
			hv = 64
			Config["h_dims"] = [hl, ha, hv]
			Config["sentisize"] = 128
			Config["shapes"] = 32
			Config["drop"] = 0.9
			NNlConfig = dict()
			NNlConfig["shapes"] = 200
			NNlConfig["drop"] = 0.9
			NNaConfig = dict()
			NNaConfig["shapes"] = 32
			NNaConfig["drop"] = 0.0
			NNvConfig = dict()
			NNvConfig["shapes"] = 100
			NNvConfig["drop"] = 0.9
			OutConfig = dict()
			OutConfig["shapes"] = 64
			OutConfig["drop"] = 0.3
			configs = [Config, NNlConfig, NNaConfig, NNvConfig, OutConfig]
			print(configs)

			models_train(X_train, y_train, X_valid, y_valid, X_test, y_test, configs, args)

	else:
		models_test(X_test, y_test, args)
