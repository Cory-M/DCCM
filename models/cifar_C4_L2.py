import torch
import torch.nn as nn
import pdb

__all__ = ['cifar_C4_L2']

class Cifar_C4_L2(nn.Module):
	def __init__(self, num_classes):
		super(Cifar_C4_L2, self).__init__()
		self.num_classes = num_classes
		
		self.features = nn.Sequential(
			  nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=0),
			  nn.BatchNorm2d(64, eps=1e-05, track_running_stats=False),
			  nn.ReLU(inplace=True),

			  nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
			  nn.BatchNorm2d(64, eps=1e-05, track_running_stats=False),
			  nn.ReLU(inplace=True),
			  nn.MaxPool2d(kernel_size=2, stride=2),
			  
			  nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=0),
			  nn.BatchNorm2d(128, eps=1e-05, track_running_stats=False),
			  nn.ReLU(inplace=True),
			  nn.MaxPool2d(kernel_size=2, stride=2),

			  nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=0),
			  nn.BatchNorm2d(256, eps=1e-05, track_running_stats=False),
			  nn.ReLU(inplace=True),
			  nn.AvgPool2d(kernel_size=4, stride=4))
	
		self.fc_layer = nn.Sequential(
			  nn.Linear(256, 64),
			  nn.BatchNorm1d(64, eps=1e-05, track_running_stats=False),
			  nn.ReLU(inplace=True),

			  nn.Linear(64, num_classes),
			  nn.BatchNorm1d(num_classes, eps=1e-05, track_running_stats=False),

			  nn.Softmax(dim=1))

		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight.data)
				m.bias.data.zero_()
			elif isinstance(m, nn.BatchNorm2d):
				m.weight.data.fill_(1)
				m.bias.data.zero_()
			elif isinstance(m, nn.BatchNorm1d):
				m.weight.data.fill_(1)
				m.bias.data.zero_()
			elif isinstance(m, nn.Linear):
				torch.nn.init.eye_(m.weight)
				m.bias.data.zero_()
	
	def forward(self, x):
		x = self.features(x)
		
		x = x.view(x.size(0), -1)
		x = self.fc_layer(x)
		
		return x

def cifar_C4_L2(num_classes):
	model = Cifar_C4_L2(num_classes)
	return model


if __name__ == '__main__':
	model = cifar_C4_L2(num_classes=10)
	data = torch.rand(128, 3, 32, 32)
	output = model(data)
	print(output.shape) 

