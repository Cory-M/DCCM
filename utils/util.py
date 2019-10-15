import os
import pickle
import logging
import shutil
import numpy as np
import torch
import torch.nn as nn
from sklearn.utils.linear_assignment_ import linear_assignment
import pdb

class AverageMeter(object):
	"""Computes and stores the average and current value"""
	def __init__(self, length=0):
		self.length = length
		self.reset()

	def reset(self):
		if self.length > 0:
			self.history = []
		else:
			self.count = 0
			self.sum = 0.0
		self.val = 0.0
		self.avg = 0.0

	def update(self, val):
		if self.length > 0:
			self.history.append(val)
			if len(self.history) > self.length:
				del self.history[0]
			self.val = self.history[-1]
			self.avg = np.mean(self.history)
		else:
			self.val = val
			self.sum += val
			self.count += 1
			self.avg = self.sum / self.count

def learning_rate_decay(optimizer, t, lr_0):
	for param_group in optimizer.param_groups:
		lr = lr_0 / np.sqrt(1 + lr_0 * param_group['weight_decay'] * t)
		param_group['lr'] = lr


class Logger():
	""" Class to update every epoch to keep trace of the results
	Methods:
		- log() log and save
	"""

	def __init__(self, path):
		self.path = path
		self.data = []

	def log(self, train_point):
		self.data.append(train_point)
		with open(os.path.join(self.path), 'wb') as fp:
			pickle.dump(self.data, fp, -1)

def create_logger(name, log_file, rank=0, level=logging.INFO):
	l = logging.getLogger(name)
	formatter = logging.Formatter('[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s][rank:{}] %(message)s'.format(rank))
	fh = logging.FileHandler(log_file)
	fh.setFormatter(formatter)
	sh = logging.StreamHandler()
	sh.setFormatter(formatter)
	l.setLevel(level)
	l.addHandler(fh)
	l.addHandler(sh)
	return l

def clustering_acc(y_true, y_pred):
	y_true = y_true.astype(np.int64)
	assert y_pred.size == y_true.size
	D = max(y_pred.max(), y_true.max()) + 1
	w = np.zeros((D, D), dtype=np.int64)
	for i in range(y_pred.size):
		w[y_pred[i], y_true[i]] += 1
	ind = linear_assignment(w.max() - w)

	return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size

	
class WeightedBCE(nn.Module):

	def __init__(self, eps=1e-12, use_gpu=True):
		super(WeightedBCE, self).__init__()
		self.eps = eps
		self.use_gpu = use_gpu

	def forward(self, inputs, targets, weights):
		log_probs_pos = torch.log(inputs + self.eps)
		log_probs_neg = torch.log(1 - inputs + self.eps)
		loss1 = - targets * log_probs_pos
		loss2 = -(1 - targets) * log_probs_neg
		loss3 = loss1 + loss2
		loss4 = loss3.mean(1)
		loss5 = weights * loss4
		loss = loss5.mean()		
	
		return loss

def load_checkpoint(model, dim_loss, classifier, optimizer, ckpt_path):

	checkpoint = torch.load(ckpt_path)
	
	model.load_state_dict(checkpoint['model'])
	dim_loss.load_state_dict(checkpoint['dim_loss'])
	optimizer.load_state_dict(checkpoint['optimizer'])

	best_nmi = checkpoint['best_nmi']
	start_epoch = checkpoint['epoch']

	return start_epoch, best_nmi


def save_checkpoint(state, is_best_nmi, filename):
	torch.save(state, filename+'.pth.tar')
	if is_best_nmi:
		shutil.copyfile(filename+'.pth.tar', filename+'_best_nmi.pth.tar')
	
def accuracy(output, target, topk=(1,)):
	"""Computes the precision@k for the specified values of k"""
	maxk = max(topk)
	batch_size = target.size(0)

	_, pred = output.topk(maxk, 1, True, True)
	pred = pred.t()
	correct = pred.eq(target.view(1, -1).expand_as(pred))

	res = []
	for k in topk:
		correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
		res.append(correct_k.mul_(100.0 / batch_size))
	return res

