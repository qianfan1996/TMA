# -*-coding:utf-8-*-
import numpy as np
import argparse
from math import ceil
from tqdm import trange
from visdom import Visdom

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, f1_score

from load_data import load_mosi
from models import MFN, TMAN1, MARN, TMAN2


def set_random_seed(seed=666):
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)


def models_train(X_train, y_train, X_valid, y_valid, X_test, y_test, configs, args):
	# viz = Visdom()
	# viz.line([[0., 0.]], [0], win='loss', opts=dict(title='train&valid loss', legend=['train loss', 'valid loss']))

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
			batch_y = torch.Tensor(y_train[start:end]).cuda()
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
			batch_y = torch.Tensor(y_valid).cuda()
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
		# viz.line([[train_loss, valid_loss]], [epoch], win='loss', update='append')
		if valid_loss <= best_valid_loss:
			best_valid_loss = valid_loss
			print(str(epoch)+'th epoch', ', train loss is', train_loss, ', valid loss is', valid_loss)
			print('Saving model...')
			torch.save(model, 'saved_models/' + args.model + '_mosi.pt')
		else:
			print(str(epoch) + 'th epoch', ', train loss is', train_loss, ', valid loss is', valid_loss)


	model = torch.load('saved_models/' + args.model + '_mosi.pt')

	predictions = predict(model, X_test)
	mae = np.mean(np.absolute(predictions-y_test))
	print("MAE:", mae)
	corr = np.corrcoef(predictions,y_test)[0][1]
	print("corr:", corr)
	mult = round(sum(np.round(predictions)==np.round(y_test))/float(len(y_test)), 5)
	print("Mult Accuracy:", mult)
	true_label = (y_test >= 0)
	predicted_label = (predictions >= 0)
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

	model = torch.load('saved_models/' + args.model + '_mosi.pt')
	print(model)

	predictions = predict(model, X_test)
	mae = np.mean(np.absolute(predictions - y_test))
	print("MAE:", mae)
	corr = np.corrcoef(predictions, y_test)[0][1]
	print("corr:", corr)
	mult = round(sum(np.round(predictions) == np.round(y_test)) / float(len(y_test)), 5)
	print("Mult Accuracy:", mult)
	true_label = (y_test >= 0)
	predicted_label = (predictions >= 0)
	print("Confusion Matrix:")
	print(confusion_matrix(true_label, predicted_label))
	print("Accuracy:", accuracy_score(true_label, predicted_label))
	f1 = round(f1_score(true_label, predicted_label, average='binary'), 5)
	print("F1 score:", f1)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("-m", "--model", choices=["mfn", "tman1", "marn", "tman2"], type=str, default="tman1", help="choose models")
	parser.add_argument("-t", "--train", action="store_false", help="train or evaluate models")
	# parser.add_argument("-d", "--dataset", choices=["mosi", "mmmo", "youtube"], type=str, default="mosi", help="choose dataset")
	parser.add_argument("--lr", type=int, default=0.001, help="learning_rate")
	parser.add_argument("--bs", type=int, default=32, help="batch_size")
	parser.add_argument("--epoch", type=int, default=50, help="num_epoch")

	args = parser.parse_args()

	X_train, y_train, X_valid, y_valid, X_test, y_test = load_mosi()

	if args.train:
		if args.model == "mfn":
			Config = dict()
			Config["input_dims"] = [300, 5, 20]
			hl = 128
			ha = 16
			hv = 64
			Config["h_dims"] = [hl, ha, hv]
			Config["memsize"] = 100
			Config["shapes"] = 200
			Config["drop"] = 0.9
			NNpConfig = dict()
			NNpConfig["shapes"] = 200
			NNpConfig["drop"] = 0.7
			Gamma1Config = dict()
			Gamma1Config["shapes"] = 50
			Gamma1Config["drop"] = 0.0
			Gamma2Config = dict()
			Gamma2Config["shapes"] = 100
			Gamma2Config["drop"] = 0.9
			OutConfig = dict()
			OutConfig["shapes"] = 64
			OutConfig["drop"] = 0.7
			configs = [Config, NNpConfig, Gamma1Config, Gamma2Config, OutConfig]
			print(configs)

			models_train(X_train, y_train, X_valid, y_valid, X_test, y_test, configs, args)

		elif args.model == "tman1":
			Config = dict()
			Config["input_dims"] = [300, 5, 20]
			hl = 128
			ha = 16
			hv = 64
			Config["h_dims"] = [hl, ha, hv]
			Config["memsize"] = 100
			Config["sentisize"] = 32
			Config["shapes"] = 64
			Config["drop"] = 0.9
			NNlConfig = dict()
			NNlConfig["shapes"] = 32
			NNlConfig["drop"] = 0.9
			NNaConfig = dict()
			NNaConfig["shapes"] = 32
			NNaConfig["drop"] = 0.3
			NNvConfig = dict()
			NNvConfig["shapes"] = 50
			NNvConfig["drop"] = 0.3
			NNpConfig = dict()
			NNpConfig["shapes"] = 32
			NNpConfig["drop"] = 0.9
			Gamma1Config = dict()
			Gamma1Config["shapes"] = 64
			Gamma1Config["drop"] = 0.5
			Gamma2Config = dict()
			Gamma2Config["shapes"] = 100
			Gamma2Config["drop"] = 0.5
			OutConfig = dict()
			OutConfig["shapes"] = 64
			OutConfig["drop"] = 0.3
			configs = [Config, NNlConfig, NNaConfig, NNvConfig, NNpConfig, Gamma1Config, Gamma2Config, OutConfig]
			print(configs)

			models_train(X_train, y_train, X_valid, y_valid, X_test, y_test, configs, args)

		elif args.model == "marn":
			Config = dict()
			Config["input_dims"] = [300, 5, 20]
			hl = 128
			ha = 16
			hv = 64
			Config["h_dims"] = [hl, ha, hv]
			Config['num_atts'] = 4
			ReduceDimConfig = dict()
			r_hl = 16
			r_ha = 128
			r_hv = 100
			ReduceDimConfig["h_dims"] = [r_hl, r_ha, r_hv]
			MapNNConfig = dict()
			MapNNConfig["shapes"] = 32
			MapNNConfig["drop"] = 0.0
			OutConfig = dict()
			OutConfig["shapes"] = 64
			OutConfig["drop"] = 0.3
			configs = [Config, ReduceDimConfig, MapNNConfig, OutConfig]
			print(configs)

			models_train(X_train, y_train, X_valid, y_valid, X_test, y_test, configs, args)

		else:
			Config = dict()
			Config["input_dims"] = [300, 5, 20]
			hl = 128
			ha = 16
			hv = 64
			Config["h_dims"] = [hl, ha, hv]
			Config["sentisize"] = 128
			Config["shapes"] = 256
			Config["drop"] = 0.3
			NNlConfig = dict()
			NNlConfig["shapes"] = 128
			NNlConfig["drop"] = 0.9
			NNaConfig = dict()
			NNaConfig["shapes"] = 200
			NNaConfig["drop"] = 0.5
			NNvConfig = dict()
			NNvConfig["shapes"] = 100
			NNvConfig["drop"] = 0.5
			OutConfig = dict()
			OutConfig["shapes"] = 64
			OutConfig["drop"] = 0.0
			configs = [Config, NNlConfig, NNaConfig, NNvConfig, OutConfig]
			print(configs)

			models_train(X_train, y_train, X_valid, y_valid, X_test, y_test, configs, args)

	else:
		models_test(X_test, y_test, args)