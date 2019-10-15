import torch
import torch.nn as nn
import pdb

def forward(model, x, _layer=None, c_layer=-1):
	layer = _layer.copy()
	# evaluation mode: return the global vector
	if layer is None:
		return model(x)
	result = list()
	c_vec = None
	
	# sobel pre-processing
	if hasattr(model, 'sobel') and model.sobel is not None:
		x = model.sobel(x)
	
	count = 1
	if not hasattr(model, 'features') or not hasattr(model, 'fc_layer'):
		raise Exception('Not Implemented Error: unsupported model type')

	# feature map from conv layers
	for m in model.features.modules():
		if not isinstance(m, nn.Sequential):
			x = m(x)
			if layer:
				if count == layer[0]:
					result.append(x)
					layer.pop(0)
			if count == c_layer:
				c_vec = x
			count += 1

	# feature vector from fc layers 
	x = x.view(x.size(0), -1)
	for m in model.fc_layer.modules():
		if not isinstance(m, nn.Sequential):
			x = m(x)
			if layer:
				if count == layer[0]:
					result.append(x)
					layer.pop(0)
			if count == c_layer:
				c_vec = x
			count += 1
	
	if len(layer) > 0:
		raise Exception('layer index is out of range')
	if count <= c_layer:
		raise Exception('c_layer index is out of range')
	return x, result, c_vec

def get_dim(model, x, _layer=None, c_layer=-1):
	layer = _layer.copy()
	result = dict()
	
	# get the size of global feature vector
	if layer is None:
		result['V_channels'] = model(x).size(1)
	
	# get the size of classifier input
	if c_layer == -1:
		result['c_size'] = result['V_channels']

	# sobel pre-processing
	if hasattr(model, 'sobel') and model.sobel is not None:
		x = model.sobel(x)
	
	count = 1
	if not hasattr(model, 'features') or not hasattr(model, 'fc_layer'):
		raise Exception('Not Implemented Error: unsupported model type')

	# feature map from conv layers
	for m in model.features.modules():
		if not isinstance(m, nn.Sequential):
			x = m(x)
			if layer:
				if count == layer[0]:
					if 'M_channels' in result:	
						raise Exception('Multiple feature-maps is not implemented')
					result['M_channels'] = x.size(1)
					result['M_size'] = (x.size(2), x.size(3))
					layer.pop(0)
			if count == c_layer:
				result['c_size'] = (x.size(1), x.size(2), x.size(3))
			count += 1
	x = x.view(x.size(0), -1)
	# feature vector from fc layer
	for m in model.fc_layer.modules():
		if not isinstance(m, nn.Sequential):
			x = m(x)
			if layer:
				if count == layer[0]:
					if 'V_channels' in result:
						raise Exception('Multiple feature-vec is not implemented')
					result['V_channels'] = x.size(1)
					layer.pop(0)
			if count == c_layer:
				result['c_size'] = x.size(1)
			count += 1
	
	if len(layer) > 0:
		raise Exception('layer index is out of range')
	if count < c_layer:
		raise Exception('c_layer index is out of range')
	return result


def comp_simi(inputs):
	"""Compute Similarity
	"""
	values, indices = torch.max(inputs, 1)
	thres = 0.9
	weights = values.ge(thres)
	weights = weights.type(torch.cuda.FloatTensor)
	[batch_size, dim] = inputs.shape
	indices = torch.unsqueeze(indices.cpu(), 1)
	one_hot_labels = torch.zeros(batch_size, dim).scatter_(1, indices, 1)
	one_hot_labels = one_hot_labels.cuda()
	inputs2 = torch.mul(inputs, inputs)
	norm2 = torch.sum(inputs2, 1)
	root_inv = torch.rsqrt(norm2)
	tmp_var1 = root_inv.expand(dim,batch_size)
	tmp_var2 = torch.t(tmp_var1)
	nml_inputs = torch.mul(inputs, tmp_var2)
	similarity = torch.matmul(nml_inputs, torch.t(nml_inputs))			
	similarity2 = similarity - torch.eye(batch_size).cuda()
	return similarity, one_hot_labels, weights

