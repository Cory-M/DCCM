import torch
import numpy as np
import random

torch.manual_seed(1234)
torch.cuda.manual_seed_all(1234)
np.random.seed(1234)
random.seed(1234)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

import argparse
import os
import pickle
import time
import itertools
import pdb
import logging
from tensorboardX import SummaryWriter
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics.cluster import adjusted_rand_score
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from keras.preprocessing.image import ImageDataGenerator
import yaml

import models
from utils.util import create_logger, AverageMeter, Logger, clustering_acc, WeightedBCE, accuracy, save_checkpoint, load_checkpoint
from utils.sampling import get_pair
from utils.mc_dataset import McDataset
from utils.functions import get_dim, forward, comp_simi 

# argparser
parser = argparse.ArgumentParser(description='PyTorch Implementation of DCCM')
parser.add_argument('--resume', default=None, type=str, help='resume from a checkpoint')
parser.add_argument('--config', default='cfgs/config.yaml', help='set configuration file')
parser.add_argument('--small_bs', default=32, type=int)
parser.add_argument('--input_size', default=96, type=int)
parser.add_argument('--split', default=None, type=int, help='divide the large forward batch to avoid OOM')

args = parser.parse_args()
with open(args.config) as f:
	config = yaml.load(f)
for k, v in config['common'].items():
	setattr(args, k, v)
coeff = config['coeff']

best_nmi = 0
start_epoch = 0

def main():
	global args, best_nmi, start_epoch
	
	os.makedirs('{}'.format(args.save_path), exist_ok=True)

	# logging configuration
	logger = create_logger('global_logger', log_file=os.path.join(args.save_path,'log.txt'))
	logger.info('{}'.format(args))
	logger.info('{}'.format(coeff))
	tb_logger = SummaryWriter(args.save_path)

	# Construct Networks (Encoder, dim_loss)
	model = models.__dict__[args.arch](args.num_classes).cuda()
	print("=> created encoder '{}'".format(args.arch))
	
	toy_input = torch.zeros([5, 3, args.input_size, args.input_size]).cuda()
	arch_info = get_dim(model, toy_input, args.layers, args.c_layer)

	dim_loss = models.__dict__['DIM_Loss'](arch_info).cuda()

	# optimizer
	para_dict = itertools.chain(filter(lambda x: x.requires_grad, model.parameters()),
		  filter(lambda x: x.requires_grad, dim_loss.parameters()))
	optimizer = torch.optim.RMSprop(para_dict, lr=args.lr, alpha=0.9)

	# criterions
	crit_graph = nn.BCELoss().cuda()
	crit_label = WeightedBCE().cuda()
	crit_c = nn.CrossEntropyLoss().cuda()

	# optionally resume from a checkpoint
	if args.resume:
		logger.info("=> loading checkpoint '{}'".format(args.resume))
		start_epoch, best_nmi = load_checkpoint(model, dim_loss, optimizer, args.resume)

	# data loading
	dataset = McDataset(
		  args.root, 
		  args.source, 
		  transform=transforms.ToTensor())
	dataloader = torch.utils.data.DataLoader(
		  dataset, batch_size=args.large_bs,
		  num_workers=args.workers, pin_memory=True, shuffle=True)
	datagen = ImageDataGenerator(
		  rotation_range=20,
		  width_shift_range=0.18,
		  height_shift_range=0.18,
		  channel_shift_range=0.1,
		  horizontal_flip=True,
		  rescale=0.95,
		  zoom_range=[0.85,1.15])

	
	for epoch in range(start_epoch, args.epochs):
	
		end = time.time()

		# Evaluation
		nmi, acc, ari = test(dataloader, model, epoch, tb_logger)
	
		# saving checkpoint
		is_best_nmi = nmi > best_nmi
		best_nmi = max(nmi, best_nmi)
		save_checkpoint({
			  'epoch': epoch, 
			  'model': model.state_dict(), 
			  'dim_loss': dim_loss.state_dict(), 
			  'best_nmi': best_nmi,
			  'optimizer': optimizer.state_dict()}, 
			  is_best_nmi, args.save_path + '/ckpt') 

		# training
		train(dataloader, model, dim_loss, crit_label, crit_graph, crit_c, optimizer, epoch, datagen, tb_logger)



def train(loader, model, dim_loss, crit_label, crit_graph, crit_c, optimizer, epoch, datagen, tb_logger):
		
	freq = args.print_freq

	batch_time = AverageMeter(freq)
	data_time = AverageMeter(freq)
	losses = AverageMeter(freq)
	g_losses = AverageMeter(freq)
	l_losses = AverageMeter(freq)
	loc_losses = AverageMeter(freq)
	
	logger = logging.getLogger('global_logger')

	# switch to train mode
	model.train()
	dim_loss.train()

	index_loc = np.arange(args.large_bs)
	end = time.time()

	for i, (input_tensor, target) in enumerate(loader):
		data_time.update(time.time() - end)
		input_var = torch.autograd.Variable(input_tensor.cuda())
		target = target.cuda()

		with torch.no_grad():
			if args.split:
				vec_list = []
				bs = args.large_bs // args.split
				for kk in range(args.split):
					temp, _, _ = forward(model, input_var[kk*bs:(kk+1)*bs], 
						  args.layers, args.c_layer)
					vec_list.append(temp)
				vec = torch.cat(vec_list, dim=0)
			else:
				vec, _, _ = forward(model, input_var, args.layers, args.c_layer)

		similarity, labels, weights = comp_simi(vec)
		mask = similarity.ge(args.thresh)

		for k in range(args.repeated):
			np.random.shuffle(index_loc)
			for j in range(similarity.shape[0] // args.small_bs):
				address = index_loc[np.arange(j*args.small_bs,(j+1)*args.small_bs)]
				input_bs = input_tensor[address]
				gt_target = target[address]
				input_bs = input_bs.numpy()

				out_target = labels[address]
				out_target = out_target.detach()
				mask_target = mask[address,:][:,address].float()
				weights_batch = weights[address]

				sign = 0
				for X_batch_i in datagen.flow(input_bs,batch_size=args.small_bs,shuffle=False):
					aug_input_bs = torch.from_numpy(X_batch_i)
					aug_input_bs = aug_input_bs.float()
					aug_input_batch_var = torch.autograd.Variable(aug_input_bs.cuda())
					vec, [M,Y], c_vec = forward(model, aug_input_batch_var, 
						  args.layers, args.c_layer)

					simi_batch, labels_batch, weigths_tmp = comp_simi(vec)
					simi_batch = simi_batch/torch.max(simi_batch)
				
					# loss computing
					Y_aug, M, M_fake = get_pair(Y, M, mask_target)
					_local = dim_loss(Y_aug, M, M_fake)

					_label = crit_label(vec, out_target, weights_batch)
					_graph = crit_graph(simi_batch, mask_target)

					loss = coeff['label'] * _label + coeff['graph'] * _graph \
						   + coeff['local'] * _local

					# records
					losses.update(loss.item())
					g_losses.update(_graph.item())
					l_losses.update(_label.item())
					loc_losses.update(_local.item())

					# compute gradient and do SGD step
					optimizer.zero_grad()
					loss.backward(retain_graph=True)
					optimizer.step()
		
					sign += 1
					if sign > 1:
						break

					# measure elapsed time
					batch_time.update(time.time() - end)
					end = time.time()

		if i % args.print_freq == 0:	
			step = epoch * len(loader) + i
			tb_logger.add_scalar('loss', losses.avg, step)
			logger.info('Epoch: [{0}/{1}][{2}/{3}]\t'
				  'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
				  'Data: {data_time.val:.3f} ({data_time.avg:.3f})\t'
				  'Loss: {loss.avg:.4f}\t'
				  'graph: {g_losses.avg:.4f}\t'
				  'label: {l_losses.avg:.4f}\t'
				  'local: {loc_losses.avg:.4f}\t'.format(
					epoch, args.epochs, i, len(loader), 
					batch_time=batch_time,
					data_time=data_time, loss=losses,
					g_losses=g_losses, l_losses=l_losses, loc_losses=loc_losses,
					))

def test(loader, model, epoch, tb_logger):
	logger = logging.getLogger('global_logger')

	model.eval()

	# Forward and save predicted labels
	gnd_labels = []
	pred_labels = []
	for i, (input_tensor, target) in enumerate(loader):
		input_var = torch.autograd.Variable(input_tensor.cuda())
		with torch.no_grad():
			if args.split:
				vec_list = []
				bs = args.large_bs // args.split
				for kk in range(args.split):
					temp, _, _ = forward(model, input_var[kk*bs:(kk+1)*bs], 
						  args.layers, args.c_layer)
					vec_list.append(temp)
				vec = torch.cat(vec_list, dim=0)
			else:
				vec, _, _ = forward(model, input_var, args.layers, args.c_layer)

		_, indices = torch.max(vec, 1)
		gnd_labels.extend(target.data.numpy())
		pred_labels.extend(indices.data.cpu().numpy())

	# Computing Evaluations
	gnd_labels = np.array(gnd_labels)
	pred_labels = np.array(pred_labels)
	
	nmi = normalized_mutual_info_score(gnd_labels, pred_labels)
	acc = clustering_acc(gnd_labels, pred_labels)
	ari = adjusted_rand_score(gnd_labels, pred_labels)

	# Logging
	logger.info('Epoch: [{0}/{1}]\t ARI against ground truth label: {2:.3f}'.format(epoch, args.epochs, ari))
	logger.info('Epoch: [{0}/{1}]\t NMI against ground truth label: {2:.3f}'.format(epoch, args.epochs, nmi)) 
	logger.info('Epoch: [{0}/{1}]\t ACC against ground truth label: {2:.3f}'.format(epoch, args.epochs, acc)) 
	step = epoch * len(loader)
	tb_logger.add_scalar('ARI', ari, step)
	tb_logger.add_scalar('NMI', nmi, step)
	tb_logger.add_scalar('ACC', acc, step)

	return nmi, acc, ari





if __name__ == '__main__':
	main()
