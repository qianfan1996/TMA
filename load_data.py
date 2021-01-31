# -*-coding:utf-8-*-

import h5py
import pickle
import numpy as np


def load_h5(file_path, key='data'):
	with h5py.File(file_path, 'r') as file:
		data = file[key][:]

	return data

def load_p(file_path):
	with open(file_path, 'rb') as file:
		data = pickle.load(file, encoding='iso-8859-1')

	return data


def load_mosi():
	X_train = load_h5('data/CMU-MOSI/X_train.h5')
	y_train = load_h5('data/CMU-MOSI/y_train.h5')

	X_valid = load_h5('data/CMU-MOSI/X_valid.h5')
	y_valid = load_h5('data/CMU-MOSI/y_valid.h5')

	X_test = load_h5('data/CMU-MOSI/X_test.h5')
	y_test = load_h5('data/CMU-MOSI/y_test.h5')

	return X_train, y_train, X_valid, y_valid, X_test, y_test


def load_mmmo():
	text_train = load_p('data/MMMO/text_train.p')
	audio_train = load_p('data/MMMO/covarep_train.p')
	video_train = load_p('data/MMMO/facet_train.p')
	X_train = np.concatenate((text_train, audio_train, video_train), axis=-1)
	y_train = load_p('data/MMMO/y_train.p')

	text_valid = load_p('data/MMMO/text_valid.p')
	audio_valid = load_p('data/MMMO/covarep_valid.p')
	video_valid = load_p('data/MMMO/facet_valid.p')
	X_valid = np.concatenate((text_valid, audio_valid, video_valid), axis=-1)
	y_valid = load_p('data/MMMO/y_valid.p')

	text_test = load_p('data/MMMO/text_test.p')
	audio_test = load_p('data/MMMO/covarep_test.p')
	video_test = load_p('data/MMMO/facet_test.p')
	X_test = np.concatenate((text_test, audio_test, video_test), axis=-1)
	y_test = load_p('data/MMMO/y_test.p')

	return X_train, y_train, X_valid, y_valid, X_test, y_test


def load_youtube():
	text_train = load_p('data/YouTube/text_train.p')
	audio_train = load_p('data/YouTube/covarep_train.p')
	video_train = load_p('data/YouTube/facet_train.p')
	X_train = np.concatenate((text_train, audio_train, video_train), axis=-1)
	y_train = load_p('data/YouTube/y_train.p')

	text_valid = load_p('data/YouTube/text_valid.p')
	audio_valid = load_p('data/YouTube/covarep_valid.p')
	video_valid = load_p('data/YouTube/facet_valid.p')
	X_valid = np.concatenate((text_valid, audio_valid, video_valid), axis=-1)
	y_valid = load_p('data/YouTube/y_valid.p')

	text_test = load_p('data/YouTube/text_test.p')
	audio_test = load_p('data/YouTube/covarep_test.p')
	video_test = load_p('data/YouTube/facet_test.p')
	X_test = np.concatenate((text_test, audio_test, video_test), axis=-1)
	y_test = load_p('data/YouTube/y_test.p')

	return X_train, y_train, X_valid, y_valid, X_test, y_test
