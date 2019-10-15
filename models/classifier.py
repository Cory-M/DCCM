from __future__ import print_function
import torch
import torch.nn as nn
import pdb

__all__ = ['SimpleClassifier','fc_Classifier', 'conv_Classifier', 'dc_Classifier']

class SimpleClassifier(nn.Module):
	def __init__(self, config):
		super(SimpleClassifier, self).__init__()
		dim = config['c_size']
		self.net = nn.Sequential(
			  nn.Linear(dim, 2000),
			  nn.Dropout(0.1),
			  nn.BatchNorm1d(2000, eps=1e-05, momentum=0.1, affine=True),
			  nn.ReLU(inplace=True),
			  nn.Linear(2000, 1000))

	def forward(self, z):
		return self.net(z)

class fc_Classifier(nn.Module):
	def __init__(self, option):
		super(fc_Classifier, self).__init__()

		self.net = nn.Sequential(
			  nn.Linear(1024, 200, bias=True),
			  nn.Dropout(0.1),
			  nn.BatchNorm1d(200, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
			  nn.ReLU(inplace=True),
			  nn.Linear(200, option['class_num'], bias=True))
	def forward(self, z):
		return self.net(z)


class conv_Classifier(nn.Module):
	def __init__(self, option):
		super(conv_Classifier, self).__init__()

		self.net = nn.Sequential(
			  nn.Linear(4096, 200, bias=True),
			  nn.Dropout(0.1),
			  nn.BatchNorm1d(200, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
			  nn.ReLU(inplace=True),
			  nn.Linear(200, option['class_num'], bias=True))

	def forward(self, z):
		return self.net(z)

class dc_Classifier(nn.Module):
	def __init__(self, conv, num_labels=1000):
		super(dc_Classifier, self).__init__()
		self.conv = conv
		
		if conv==1:
			self.av_pool = nn.AvgPool2d(6, stride=6, padding=3)
			s = 9600
		elif conv==2:
			self.av_pool = nn.AvgPool2d(4, stride=4, padding=0)
			s = 9216
		elif conv==3:
			self.av_pool = nn.AvgPool2d(3, stride=3, padding=1)
			s = 9600
		elif conv==4:
			self.av_pool = nn.AvgPool2d(3, stride=3, padding=1)
			s = 9600
		elif conv==5:
			self.av_pool = nn.AvgPool2d(2, stride=2, padding=0)
			s = 9216
		self.linear = nn.Linear(s, num_labels)
	
	def forward(self, x):
		x = self.av_pool(x)
		x = x.view(x.size(0), x.size(1) * x.size(2) * x.size(3))
		return self.linear(x)

