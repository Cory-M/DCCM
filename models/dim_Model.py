#-----------------------------------------------------------------------------
# model definition
#-----------------------------------------------------------------------------
import pdb
import torch, torchvision, torch.nn.functional as F
from torch.autograd import Variable
import torch.nn as nn
__all__ = ['DIM_Loss']


class LocalDiscriminator(torch.nn.Module):
	r"""
	the local discriminator with architecture described in
	Figure 4 and Table 6 in appendix 1A of https://arxiv.org/pdf/1808.06670.pdf.
	input is the concatenate of
	"replicated feature vector E (with M_shape now)" + "M"

	replicated means that all pixels are the same, they are just copies.
	"""
	def __init__(self, M_channels=128, V_channels=64, interm_channels=512):
		super().__init__()

		in_channels = V_channels + M_channels
		self.c0 = torch.nn.Conv2d(in_channels, interm_channels, 
			  kernel_size=1, stride=1, bias=False)
		self.c1 = torch.nn.Conv2d(interm_channels, interm_channels, 
			  kernel_size=1, stride=1, bias=False)
		self.c2 = torch.nn.Conv2d(interm_channels, 1, 
			  kernel_size=1, stride=1, bias=False)

	def forward(self, x):

		score = F.relu(self.c0(x))
		score = F.relu(self.c1(score))
		score = self.c2(score)

		return score


class DIM_Loss(torch.nn.Module):

	def __init__(self, config, **kwargs):
		super().__init__()

		self.get_models(M_channels=config['M_channels'], 
			  V_channels=config['V_channels'], **kwargs)
		self.M_size = config['M_size']


	def get_models(self, M_channels, V_channels, interm_channels_L=512):

		self.local_D = LocalDiscriminator(M_channels, V_channels, interm_channels_L)

	def forward(self, Y, M, M_fake):

		Y_replicated = Y.unsqueeze(-1).unsqueeze(-1)
		Y_replicated = Y_replicated.expand(-1, -1, self.M_size[0], self.M_size[1])
		
		Y_cat_M = torch.cat((M, Y_replicated), dim=1)
		Y_cat_M_fake = torch.cat((M_fake, Y_replicated), dim=1)

		# local loss
		Ej = -F.softplus(-self.local_D(Y_cat_M)).mean()
		Em = F.softplus(self.local_D(Y_cat_M_fake)).mean()
		local_loss = -(Ej - Em)

		return local_loss
