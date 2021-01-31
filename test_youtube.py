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

from sklearn.metrics import accuracy_score, f1_score

from load_data import load_youtube
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

	criterion = nn.CrossEntropyLoss()
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
			batch_y = torch.Tensor(np.argmax(y_train[start:end], 1)).long().cuda()
			predictions = model.forward(batch_X)
			loss = criterion(predictions, batch_y)
			loss.backward()
			optimizer.step()
			epoch_loss += loss.item()
		return epoch_loss / num_batches

	def validate(model, X_valid, y_valid, criterion):
		model.eval()
		with torch.no_grad():
			batch_X = torch.Tensor(X_valid).cuda()
			batch_y = torch.Tensor(np.argmax(y_valid, 1)).long().cuda()
			predictions = model.forward(batch_X)
			epoch_loss = criterion(predictions, batch_y).item()
		return epoch_loss

	def predict(model, X_test):
		model.eval()
		with torch.no_grad():
			batch_X = torch.Tensor(X_test).cuda()
			predictions = model.forward(batch_X)
			predictions = predictions.cpu().numpy()

		return predictions

	best_valid_loss = 999999.0
	rand = random.randint(0, 1000000)
	for epoch in trange(args.epoch):
		train_loss = train(model, args.bs, X_train, y_train, optimizer, criterion)
		valid_loss = validate(model, X_valid, y_valid, criterion)
		if valid_loss <= best_valid_loss:
			best_valid_loss = valid_loss
			print(str(epoch)+'th epoch', ', train loss is', train_loss, ', valid loss is', valid_loss)
			print('Saving model...')
			torch.save(model, 'saved_models/' + args.model + '_youtube.pt')
		else:
			print(str(epoch) + 'th epoch', ', train loss is', train_loss, ', valid loss is', valid_loss)

	model = torch.load('saved_models/' + args.model + '_youtube.pt')

	predictions = predict(model, X_test)
	predicted_label = np.argmax(predictions, 1)
	true_label = np.argmax(y_test, 1)
	accuracy = accuracy_score(true_label, predicted_label)
	print("Accuracy ", accuracy)
	f1 = round(f1_score(true_label, predicted_label, average='macro'), 5)
	print("F1 score:", f1)


def models_test(X_test, y_test, args):
	X_test = X_test.swapaxes(0, 1)

	def predict(model, X_test):
		model.eval()
		with torch.no_grad():
			batch_X = torch.Tensor(X_test).cuda()
			predictions = model.forward(batch_X)
			predictions = predictions.cpu().numpy()
		return predictions

	model = torch.load('saved_models/' + args.model + '_youtube.pt')

	predictions = predict(model, X_test)
	predicted_label = np.argmax(predictions, 1)
	true_label = np.argmax(y_test, 1)
	accuracy = accuracy_score(true_label, predicted_label)
	print("Accuracy ", accuracy)
	f1 = round(f1_score(true_label, predicted_label, average='macro'), 5)
	print("F1 score:", f1)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("-m", "--model", choices=["mfn", "tman1", "marn", "tman2"], type=str, default="tman1", help="choose models")
	parser.add_argument("-t", "--train", action="store_false", help="train or evaluate models")
	parser.add_argument("--lr", type=int, default=0.001, help="learning_rate")
	parser.add_argument("--bs", type=int, default=32, help="batch_size")
	parser.add_argument("--epoch", type=int, default=50, help="num_epoch")

	args = parser.parse_args()

	X_train, y_train, X_valid, y_valid, X_test, y_test = load_youtube()

	if args.train:
		if args.model == "mfn":
			Config = dict()
			Config["input_dims"] = [300, 74, 35]
			hl = 128
			ha = 64
			hv = 64
			Config["h_dims"] = [hl, ha, hv]
			Config["memsize"] = 100
			Config["shapes"] = 64
			Config["drop"] = 0.3
			NNpConfig = dict()
			NNpConfig["shapes"] = 64
			NNpConfig["drop"] = 0.5
			Gamma1Config = dict()
			Gamma1Config["shapes"] = 256
			Gamma1Config["drop"] = 0.0
			Gamma2Config = dict()
			Gamma2Config["shapes"] = 64
			Gamma2Config["drop"] = 0.7
			OutConfig = dict()
			OutConfig["shapes"] = 64
			OutConfig["drop"] = 0.5
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
			Config["sentisize"] = 32
			Config["shapes"] = 64
			Config["drop"] = 0.3
			NNlConfig = dict()
			NNlConfig["shapes"] = 128
			NNlConfig["drop"] = 0.7
			NNaConfig = dict()
			NNaConfig["shapes"] = 8
			NNaConfig["drop"] = 0.7
			NNvConfig = dict()
			NNvConfig["shapes"] = 128
			NNvConfig["drop"] = 0.0
			NNpConfig = dict()
			NNpConfig["shapes"] = 64
			NNpConfig["drop"] = 0.0
			Gamma1Config = dict()
			Gamma1Config["shapes"] = 32
			Gamma1Config["drop"] = 0.7
			Gamma2Config = dict()
			Gamma2Config["shapes"] = 128
			Gamma2Config["drop"] = 0.9
			OutConfig = dict()
			OutConfig["shapes"] = 64
			OutConfig["drop"] = 0.7
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
			r_hl = 8
			r_ha = 64
			r_hv = 80
			ReduceDimConfig["h_dims"] = [r_hl, r_ha, r_hv]
			MapNNConfig = dict()
			MapNNConfig["shapes"] = 200
			MapNNConfig["drop"] = 0.0
			OutConfig = dict()
			OutConfig["shapes"] = 64
			OutConfig["drop"] = 0.3
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
			Config["sentisize"] = 32
			Config["shapes"] = 200
			Config["drop"] = 0.3
			NNlConfig = dict()
			NNlConfig["shapes"] = 16
			NNlConfig["drop"] = 0.7
			NNaConfig = dict()
			NNaConfig["shapes"] = 128
			NNaConfig["drop"] = 0.9
			NNvConfig = dict()
			NNvConfig["shapes"] = 32
			NNvConfig["drop"] = 0.5
			OutConfig = dict()
			OutConfig["shapes"] = 64
			OutConfig["drop"] = 0.9
			configs = [Config, NNlConfig, NNaConfig, NNvConfig, OutConfig]
			print(configs)

			models_train(X_train, y_train, X_valid, y_valid, X_test, y_test, configs, args)

	else:
		models_test(X_test, y_test, args)