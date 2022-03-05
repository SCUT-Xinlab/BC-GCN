import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import config

lamda = config.lamda
SE = config.SE
node = config.node  # number of rois

class GPC(nn.Module):

	def __init__(self, in_dim, out_dim):
		super(GPC, self).__init__()
		self.out_dim = out_dim
		self.conv = nn.Conv2d(in_dim, out_dim, (1, node))
		nn.init.normal_(self.conv.weight, std=math.sqrt(2*(1-lamda)/(node*in_dim+node*out_dim)))

	def forward(self, x):
		batchsize = x.shape[0]

		x_c = self.conv(x)
		x_C = x_c.expand(batchsize, self.out_dim, node, node)
		x_R = x_C.permute(0,1,3,2)
		x = x_C+x_R

		return x

class GPC_Res(nn.Module):

	def __init__(self, in_dim, out_dim):
		super(GPC_Res, self).__init__()
		self.out_dim = out_dim
		self.conv = nn.Conv2d(in_dim, out_dim, (1, node))
		nn.init.normal_(self.conv.weight, std=math.sqrt(2*(1-lamda)/(node*in_dim+node*out_dim)))
		self.convres = nn.Conv2d(in_dim, out_dim, 1)
		nn.init.normal_(self.convres.weight, std=math.sqrt(4*lamda/(in_dim+out_dim)))

	def forward(self, x):
		batchsize = x.shape[0]

		res = self.convres(x)
		x_c = self.conv(x)
		x_C = x_c.expand(batchsize, self.out_dim, node, node)
		x_R = x_C.permute(0,1,3,2)
		x = x_C+x_R+res

		return x

class GPC_SE(nn.Module):

	def __init__(self, in_dim, out_dim):
		super(GPC_SE, self).__init__()
		self.out_dim = out_dim
		self.conv = nn.Conv2d(in_dim, out_dim, (1, node))
		nn.init.normal_(self.conv.weight, std=math.sqrt(2*(1-lamda)/(node*in_dim+node*out_dim)))
		self.convres = nn.Conv2d(in_dim, out_dim, 1)
		nn.init.normal_(self.convres.weight, std=math.sqrt(4*lamda/(in_dim+out_dim)))

		self.sed = nn.Linear(out_dim, int(out_dim/SE), False)
		self.seu = nn.Linear(int(out_dim/SE), out_dim, False)

	def forward(self, x):
		batchsize = x.shape[0]

		res = self.convres(x)
		x_c = self.conv(x)
		x_C = x_c.expand(batchsize, self.out_dim, node, node)
		x_R = x_C.permute(0,1,3,2)
		x = x_C+x_R

		se = torch.mean(x,(2,3))
		se = self.sed(se)
		se = F.relu(se)
		se = self.seu(se)
		se = torch.sigmoid(se)
		se = se.unsqueeze(2).unsqueeze(3)

		x = x.mul(se)
		x = x+res

		return x

class EP(nn.Module):

	def __init__(self, in_dim, out_dim):
		super(EP, self).__init__()
		self.conv = nn.Conv2d(in_dim, out_dim, (1, node))
		nn.init.normal_(self.conv.weight, std=math.sqrt(4/(node*in_dim+out_dim)))


	def forward(self, x):

		x = self.conv(x)

		return x

class NP(nn.Module):

	def __init__(self, in_dim, out_dim):
		super(NP, self).__init__()
		self.conv = nn.Conv2d(in_dim, out_dim, (node, 1))
		nn.init.normal_(self.conv.weight, std=math.sqrt(4/(node*in_dim+out_dim)))

	def forward(self, x):

		x = self.conv(x)

		return x

class BC_GCN(nn.Module):
	def __init__(self, GPC_dim_1, GPC_dim_2, GPC_dim_3, EP_dim, NP_dim):
		super(BC_GCN, self).__init__()

		print('Current model : BC_GCN')

		self.GPC_1 = GPC(1, GPC_dim_1)
		self.GPC_2 = GPC(GPC_dim_1, GPC_dim_2)
		self.GPC_3 = GPC(GPC_dim_2, GPC_dim_3)

		self.EP = EP(GPC_dim_3, EP_dim)

		self.NP = NP(EP_dim, NP_dim)

		self.fc = nn.Linear(NP_dim, 1)

	def forward(self, x):

		x = self.GPC_1(x)
		x = F.relu(x)

		x = self.GPC_2(x)
		x = F.relu(x)

		x = self.GPC_3(x)
		x = F.relu(x)

		x = self.EP(x)
		x = F.relu(x)

		x = self.NP(x)
		x = F.relu(x)

		x = x.view(x.size(0),-1)

		x = self.fc(x)

		return x

class BC_GCN_Res(nn.Module):
	def __init__(self, GPC_dim_1, GPC_dim_2, GPC_dim_3, EP_dim, NP_dim):
		super(BC_GCN_Res, self).__init__()

		print('Current model : BC_GCN_Res')

		self.GPC_1 = GPC_Res(1, GPC_dim_1)
		self.GPC_2 = GPC_Res(GPC_dim_1, GPC_dim_2)
		self.GPC_3 = GPC_Res(GPC_dim_2, GPC_dim_3)

		self.EP = EP(GPC_dim_3, EP_dim)

		self.NP = NP(EP_dim, NP_dim)

		self.fc = nn.Linear(NP_dim, 1)

	def forward(self, x):

		x = self.GPC_1(x)
		x = F.relu(x)

		x = self.GPC_2(x)
		x = F.relu(x)

		x = self.GPC_3(x)
		x = F.relu(x)

		x = self.EP(x)
		x = F.relu(x)

		x = self.NP(x)
		x = F.relu(x)

		x = x.view(x.size(0),-1)

		x = self.fc(x)

		return x

class BC_GCN_SE(nn.Module):
	def __init__(self, GPC_dim_1, GPC_dim_2, GPC_dim_3, EP_dim, NP_dim):
		super(BC_GCN_SE, self).__init__()

		print('Current model : BC_GCN_SE')

		self.GPC_1 = GPC_SE(1, GPC_dim_1)
		self.GPC_2 = GPC_SE(GPC_dim_1, GPC_dim_2)
		self.GPC_3 = GPC_SE(GPC_dim_2, GPC_dim_3)

		self.EP = EP(GPC_dim_3, EP_dim)

		self.NP = NP(EP_dim, NP_dim)

		self.fc = nn.Linear(NP_dim, 1)

	def forward(self, x):

		x = self.GPC_1(x)
		x = F.relu(x)

		x = self.GPC_2(x)
		x = F.relu(x)

		x = self.GPC_3(x)
		x = F.relu(x)

		x = self.EP(x)
		x = F.relu(x)

		x = self.NP(x)
		x = F.relu(x)

		x = x.view(x.size(0),-1)

		x = self.fc(x)

		return x


def weights_init(m):
	if isinstance(m, nn.Conv2d):
		#nn.init.kaiming_normal_(m.weight, mode='fan_out')
		#nn.init.xavier_uniform_(m.weight)
		nn.init.constant_(m.bias, 0)
	elif isinstance(m, nn.Linear):
		nn.init.xavier_uniform_(m.weight)
		#nn.init.constant_(m.bias, 0)
