import torch
import torch.nn as nn
import torch.nn.functional as F
import math

lamda = 0.5
SE = 4
node = 112  # the number of rois

class GCN(nn.Module):
	def __init__(self, layer1, layer2):
		super(GCN, self).__init__()

		print('Current model: GCN')

		self.fc1 = nn.Linear(node, layer1)
		self.fc2 = nn.Linear(layer1, layer2)
		self.fc3 = nn.Linear(layer2, 1)
		self.fc = nn.Linear(node, 1)

	def forward(self, x):

		batchsize = int(torch.numel(x)/node/node)

		h = torch.eye(node).expand(batchsize,node,node).cuda()

		h = torch.bmm(x,h)
		h = self.fc1(h)
		h = F.relu(h)

		h = torch.bmm(x,h)
		h = self.fc2(h)
		h = F.relu(h)

		h = torch.bmm(x,h)
		h = self.fc3(h)
		h = F.relu(h)

		h = h.view(h.size(0),-1)

		h = self.fc(h)

		return h

class CNN(nn.Module):
	def __init__(self, layer_1, layer_2, layer_3, layer_4):
		super(CNN, self).__init__()

		print('Current model: CNN')

		self.conv11 = nn.Conv2d(1, layer_1, 3)
		self.conv12 = nn.Conv2d(layer_1, layer_1, 3)

		self.conv2 = nn.Conv2d(layer_1, layer_2, 3)

		self.conv3 = nn.Conv2d(layer_2, layer_3, 3)

		self.conv4 = nn.Conv2d(layer_3, layer_4, 3)

		self.fc = nn.Linear(layer_4, 1)
		nn.init.constant_(self.fc.bias, 0)

		self.maxpool = nn.MaxPool2d(2, 2)
		self.avgpool = nn.AvgPool2d(10,10)

	def forward(self, x):

		x = self.conv11(x)
		x = F.relu(x)

		x = self.conv12(x)
		x = F.relu(x)

		x = self.maxpool(x)

		x = self.conv2(x)
		x = F.relu(x)

		x = self.maxpool(x)

		x = self.conv3(x)
		x = F.relu(x)

		x = self.maxpool(x)

		x = self.conv4(x)
		x = F.relu(x)

		x = self.avgpool(x)

		x = x.view(x.size(0),-1)

		x = self.fc(x)

		return x

class BrainNet(nn.Module):
	def __init__(self, E2E_1, E2E_2, E2E_3, E2N, N2G):
		super(BrainNet, self).__init__()

		print('Current model: BrainNet')

		self.E2E_1 = E2E_1
		self.E2E_2 = E2E_2
		self.E2E_3 = E2E_3

		self.convE1l = nn.Conv2d(1, E2E_1, (1,node))
		self.convE1r = nn.Conv2d(1, E2E_1, (node,1))

		self.convE2l = nn.Conv2d(E2E_1, E2E_2, (1,node))
		self.convE2r = nn.Conv2d(E2E_1, E2E_2, (node,1))

		self.convE3l = nn.Conv2d(E2E_2, E2E_3, (1,node))
		self.convE3r = nn.Conv2d(E2E_2, E2E_3, (node,1))

		self.convN = nn.Conv2d(E2E_3, E2N, (1,node))

		self.convG = nn.Conv2d(E2N, N2G, (node,1))

		self.fc = nn.Linear(N2G, 1)
		nn.init.constant_(self.fc.bias, 0)

	def forward(self, x):

		batchsize = int(torch.numel(x)/node/node)

		x_c = self.convE1l(x)
		x_C = x_c.expand(batchsize,self.E2E_1,node,node)
		x_r = self.convE1r(x)
		x_R = x_r.expand(batchsize,self.E2E_1,node,node)
		x = x_C+x_R
		x = F.relu(x)

		x_c = self.convE2l(x)
		x_C = x_c.expand(batchsize,self.E2E_2,node,node)
		x_r = self.convE2r(x)
		x_R = x_r.expand(batchsize,self.E2E_2,node,node)
		x = x_C+x_R
		x = F.relu(x)

		x_c = self.convE3l(x)
		x_C = x_c.expand(batchsize,self.E2E_3,node,node)
		x_r = self.convE3r(x)
		x_R = x_r.expand(batchsize,self.E2E_3,node,node)
		x = x_C+x_R
		x = F.relu(x)

		x = self.convN(x)
		x = F.relu(x)

		x = self.convG(x)
		x = F.relu(x)

		x = x.view(x.size(0),-1)

		x = self.fc(x)

		return x

class BrainNet_Res(nn.Module):
	def __init__(self, E2E_1, E2E_2, E2E_3, E2N, N2G):
		super(BrainNet_Res, self).__init__()

		print('Current model: BrainNet_Res')

		self.E2E_1 = E2E_1
		self.E2E_2 = E2E_2
		self.E2E_3 = E2E_3

		self.convE1l = nn.Conv2d(1, E2E_1, (1,node))
		self.convE1r = nn.Conv2d(1, E2E_1, (node,1))
		self.convE1res = nn.Conv2d(1, E2E_1, 1)

		self.convE2l = nn.Conv2d(E2E_1, E2E_2, (1,node))
		self.convE2r = nn.Conv2d(E2E_1, E2E_2, (node,1))
		self.convE2res = nn.Conv2d(E2E_1, E2E_2, 1)

		self.convE3l = nn.Conv2d(E2E_2, E2E_3, (1,node))
		self.convE3r = nn.Conv2d(E2E_2, E2E_3, (node,1))
		self.convE3res = nn.Conv2d(E2E_2, E2E_3, 1)

		self.convN = nn.Conv2d(E2E_3, E2N, (1,node))

		self.convG = nn.Conv2d(E2N, N2G, (node,1))

		self.fc = nn.Linear(N2G, 1)
		nn.init.constant_(self.fc.bias, 0)

	def forward(self, x):

		batchsize = int(torch.numel(x)/node/node)

		res = self.convE1res(x)
		x_c = self.convE1l(x)
		x_C = x_c.expand(batchsize,self.E2E_1,node,node)
		x_r = self.convE1r(x)
		x_R = x_r.expand(batchsize,self.E2E_1,node,node)
		x = x_C+x_R+res
		x = F.relu(x)

		res = self.convE2res(x)
		x_c = self.convE2l(x)
		x_C = x_c.expand(batchsize,self.E2E_2,node,node)
		x_r = self.convE2r(x)
		x_R = x_r.expand(batchsize,self.E2E_2,node,node)
		x = x_C+x_R+res
		x = F.relu(x)

		res = self.convE3res(x)
		x_c = self.convE3l(x)
		x_C = x_c.expand(batchsize,self.E2E_3,node,node)
		x_r = self.convE3r(x)
		x_R = x_r.expand(batchsize,self.E2E_3,node,node)
		x = x_C+x_R+res
		x = F.relu(x)

		x = self.convN(x)
		x = F.relu(x)

		x = self.convG(x)
		x = F.relu(x)

		x = x.view(x.size(0),-1)

		x = self.fc(x)

		return x

class BrainNet_SE(nn.Module):
	def __init__(self, E2E_1, E2E_2, E2E_3, E2N, N2G):
		super(BrainNet_SE, self).__init__()

		print('Current model: BrainNet_SE')

		self.E2E_1 = E2E_1
		self.E2E_2 = E2E_2
		self.E2E_3 = E2E_3

		self.convE1l = nn.Conv2d(1, E2E_1, (1,node))
		self.convE1r = nn.Conv2d(1, E2E_1, (node,1))
		self.convE1res = nn.Conv2d(1, E2E_1, 1)

		self.sed1 = nn.Linear(E2E_1, int(E2E_1/SE), False)
		self.seu1 = nn.Linear(int(E2E_1/SE), E2E_1, False)

		self.convE2l = nn.Conv2d(E2E_1, E2E_2, (1,node))
		self.convE2r = nn.Conv2d(E2E_1, E2E_2, (node,1))
		self.convE2res = nn.Conv2d(E2E_1, E2E_2, 1)

		self.sed2 = nn.Linear(E2E_2, int(E2E_2/SE), False)
		self.seu2 = nn.Linear(int(E2E_2/SE), E2E_2, False)

		self.convE3l = nn.Conv2d(E2E_2, E2E_3, (1,node))
		self.convE3r = nn.Conv2d(E2E_2, E2E_3, (node,1))
		self.convE3res = nn.Conv2d(E2E_2, E2E_3, 1)

		self.sed3 = nn.Linear(E2E_3, int(E2E_3/SE), False)
		self.seu3 = nn.Linear(int(E2E_3/SE), E2E_3, False)

		self.convN = nn.Conv2d(E2E_3, E2N, (1,node))

		self.convG = nn.Conv2d(E2N, N2G, (node,1))

		self.fc = nn.Linear(N2G, 1)
		nn.init.constant_(self.fc.bias, 0)

	def forward(self, x):

		batchsize = int(torch.numel(x)/node/node)

		res = self.convE1res(x)
		x_c = self.convE1l(x)
		x_C = x_c.expand(batchsize,self.E2E_1,node,node)
		x_r = self.convE1r(x)
		x_R = x_r.expand(batchsize,self.E2E_1,node,node)
		x = x_C+x_R

		se = torch.mean(x,(2,3))
		se = self.sed2(se)
		se = F.relu(se)
		se = self.seu2(se)
		se = torch.sigmoid(se)
		se = se.unsqueeze(2).unsqueeze(3)

		x = x.mul(se)+res
		x = F.relu(x)

		res = self.convE2res(x)
		x_c = self.convE2l(x)
		x_C = x_c.expand(batchsize,self.E2E_2,node,node)
		x_r = self.convE2r(x)
		x_R = x_r.expand(batchsize,self.E2E_2,node,node)
		x = x_C+x_R

		se = torch.mean(x,(2,3))
		se = self.sed2(se)
		se = F.relu(se)
		se = self.seu2(se)
		se = torch.sigmoid(se)
		se = se.unsqueeze(2).unsqueeze(3)

		x = x.mul(se)+res
		x = F.relu(x)

		res = self.convE3res(x)
		x_c = self.convE3l(x)
		x_C = x_c.expand(batchsize,self.E2E_3,node,node)
		x_r = self.convE3r(x)
		x_R = x_r.expand(batchsize,self.E2E_3,node,node)
		x = x_C+x_R

		se = torch.mean(x,(2,3))
		se = self.sed3(se)
		se = F.relu(se)
		se = self.seu3(se)
		se = torch.sigmoid(se)
		se = se.unsqueeze(2).unsqueeze(3)

		x = x.mul(se)+res
		x = F.relu(x)

		x = self.convN(x)
		x = F.relu(x)

		x = self.convG(x)
		x = F.relu(x)

		x = x.view(x.size(0),-1)

		x = self.fc(x)

		return x

class MLP(nn.Module):
	def __init__(self, layer_1, layer_2):
		super(MLP, self).__init__()

		print('Current model: MLP')

		self.fc1 = nn.Linear(node * node, layer_1)
		nn.init.constant_(self.fc1.bias, 0)

		self.fc2 = nn.Linear(layer_1, layer_2)
		nn.init.constant_(self.fc2.bias, 0)

		self.fc3 = nn.Linear(layer_2, 1)
		nn.init.constant_(self.fc3.bias, 0)

		# self.bn1 = nn.BatchNorm1d(layer_1)
		# self.bn2 = nn.BatchNorm1d(layer_2)
		# self.dropout = nn.Dropout(0.4)

	def forward(self, x):

		x = x.view(x.size(0), -1)

		x = self.fc1(x)
		# x = self.bn1(x)
		# x = F.relu(x)

		x = self.fc2(x)
		# x = self.bn2(x)
		# x = F.relu(x)
		# x = self.dropout(x)

		x = self.fc3(x)

		return x

class MLP_large(nn.Module):
	def __init__(self, layer_1, layer_2, layer_3):
		super(MLP_large, self).__init__()

		print('Current model: MLP_large')

		self.fc1 = nn.Linear(node * node, layer_1)
		nn.init.constant_(self.fc1.bias, 0)

		self.fc2 = nn.Linear(layer_1, layer_2)
		nn.init.constant_(self.fc2.bias, 0)

		self.fc3 = nn.Linear(layer_2, layer_3)
		nn.init.constant_(self.fc3.bias, 0)

		self.fc4 = nn.Linear(layer_3, 1)
		nn.init.constant_(self.fc4.bias, 0)

		# self.bn1 = nn.BatchNorm1d(layer_1)
		# self.bn2 = nn.BatchNorm1d(layer_2)
		# self.dropout = nn.Dropout(0.4)

	def forward(self, x):

		x = x.view(x.size(0), -1)

		x = self.fc1(x)
		# x = self.bn1(x)
		# x = F.relu(x)

		x = self.fc2(x)
		# x = self.bn2(x)
		# x = F.relu(x)
		# x = self.dropout(x)

		x = self.fc3(x)
		# x = F.relu(x)

		x = self.fc4(x)

		return x

class MLP_bn(nn.Module):
	def __init__(self, layer_1, layer_2):
		super(MLP_bn, self).__init__()

		print('Current model: MLP_bn')

		self.fc1 = nn.Linear(node * node, layer_1)
		nn.init.constant_(self.fc1.bias, 0)

		self.fc2 = nn.Linear(layer_1, layer_2)
		nn.init.constant_(self.fc2.bias, 0)

		self.fc3 = nn.Linear(layer_2, 1)
		nn.init.constant_(self.fc3.bias, 0)

		self.bn1 = nn.BatchNorm1d(layer_1)
		self.bn2 = nn.BatchNorm1d(layer_2)
		# self.dropout = nn.Dropout(0.2)

	def forward(self, x):

		x = x.view(x.size(0), -1)

		x = self.fc1(x)
		x = self.bn1(x)
		# x = F.relu(x)

		x = self.fc2(x)
		x = self.bn2(x)
		# x = F.relu(x)
		# x = self.dropout(x)

		x = self.fc3(x)

		return x

class MLP_drop(nn.Module):
	def __init__(self, layer_1, layer_2):
		super(MLP_drop, self).__init__()

		print('Current model: MLP_drop')

		self.fc1 = nn.Linear(node * node, layer_1)
		nn.init.constant_(self.fc1.bias, 0)

		self.fc2 = nn.Linear(layer_1, layer_2)
		nn.init.constant_(self.fc2.bias, 0)

		self.fc3 = nn.Linear(layer_2, 1)
		nn.init.constant_(self.fc3.bias, 0)

		# self.bn1 = nn.BatchNorm1d(layer_1)
		# self.bn2 = nn.BatchNorm1d(layer_2)
		self.dropout = nn.Dropout(0.2)

	def forward(self, x):

		x = x.view(x.size(0), -1)

		x = self.fc1(x)
		# x = self.bn1(x)
		# x = F.relu(x)

		x = self.fc2(x)
		# x = self.bn2(x)
		# x = F.relu(x)
		x = self.dropout(x)

		x = self.fc3(x)

		return x

class CNN_large(nn.Module):
	def __init__(self, layer_1, layer_2, layer_3, layer_4, layer_5):
		super(CNN_large, self).__init__()

		print('Current model: CNN_large')

		self.conv11 = nn.Conv2d(1, layer_1, 3)
		self.conv12 = nn.Conv2d(layer_1, layer_1, 3)

		self.conv2 = nn.Conv2d(layer_1, layer_2, 3)

		self.conv3 = nn.Conv2d(layer_2, layer_3, 3)

		self.conv4 = nn.Conv2d(layer_3, layer_4, 3)

		self.conv5 = nn.Conv2d(layer_4, layer_5, 3)

		self.fc = nn.Linear(layer_5, 1)
		nn.init.constant_(self.fc.bias, 0)

		self.maxpool = nn.MaxPool2d(2, 2)
		self.avgpool = nn.AvgPool2d(5,5)

	def forward(self, x):

		x = self.conv11(x)
		x = F.relu(x)

		x = self.conv12(x)
		x = F.relu(x)

		x = self.maxpool(x)

		x = self.conv2(x)
		x = F.relu(x)

		x = self.maxpool(x)

		x = self.conv3(x)
		x = F.relu(x)

		x = self.maxpool(x)

		x = self.conv4(x)
		x = F.relu(x)

		x = self.conv5(x)
		x = F.relu(x)

		x = self.avgpool(x)

		x = x.view(x.size(0),-1)

		x = self.fc(x)

		return x

class GCN_large(nn.Module):
	def __init__(self, layer1, layer2, layer3, layer4):
		super(GCN_large, self).__init__()

		print('Current model: GCN_large')

		self.fc1 = nn.Linear(node, layer1)
		self.fc2 = nn.Linear(layer1, layer2)
		self.fc3 = nn.Linear(layer2, layer3)
		self.fc4 = nn.Linear(layer3, layer4)
		self.fc5 = nn.Linear(layer4, 1)
		self.fc = nn.Linear(node, 1)

	def forward(self, x):

		batchsize = int(torch.numel(x)/node/node)

		h = torch.eye(node).expand(batchsize,node,node).cuda()

		h = torch.bmm(x,h)
		h = self.fc1(h)
		h = F.relu(h)

		h = torch.bmm(x,h)
		h = self.fc2(h)
		h = F.relu(h)

		h = torch.bmm(x,h)
		h = self.fc3(h)
		h = F.relu(h)

		h = torch.bmm(x,h)
		h = self.fc4(h)
		h = F.relu(h)

		h = torch.bmm(x,h)
		h = self.fc5(h)
		h = F.relu(h)

		h = h.view(h.size(0),-1)

		h = self.fc(h)

		return h

def weights_init(m):
	if isinstance(m, nn.Conv2d):
		#nn.init.kaiming_normal_(m.weight, mode='fan_out')
		nn.init.xavier_uniform_(m.weight)
		nn.init.constant_(m.bias, 0)
	elif isinstance(m, nn.Linear):
		nn.init.xavier_uniform_(m.weight)
		#nn.init.constant_(m.bias, 0)

class PR_GNN(nn.Module):
	def __init__(self, layer1, layer2, layer3):
		super(PR_GNN, self).__init__()

		print('Current model: PR_GNN')

		self.h = nn.Parameter(torch.eye(node), requires_grad=False)
		self.gat1 = GAT(node, layer1)
		self.pool1 = Pool(0.5, layer1)
		self.gat2 = GAT(layer1, layer2)
		self.pool2 = Pool(0.5, layer2)
		self.fc1 = nn.Linear(layer2, layer3)
		self.fc2 = nn.Linear(layer3, 1)

	def forward(self, x):

		batchsize = x.shape[0]

		h = self.h.expand(batchsize,node,node)

		h = self.gat1(x, h)
		h = F.relu(h)
		x, h = self.pool1(x, h)
		h = self.gat2(x, h)
		h = F.relu(h)
		x, h = self.pool2(x, h)
		h = torch.mean(h, 1)
		h = self.fc1(h)
		h = F.relu(h)
		h = self.fc2(h)

		return h

class BrainGNN(nn.Module):
	def __init__(self, cluster, layer1, layer2, layer3):
		super(BrainGNN, self).__init__()

		print('Current model: BrainGNN')

		self.gc1 = Ra_GC(node, cluster, node, layer1)
		self.pool1 = Pool(0.5, layer1)
		self.gc2 = Ra_GC(int(node * 0.5), cluster, layer1, layer2)
		self.pool2 = Pool(0.5, layer2)
		self.fc1 = nn.Linear(layer2, layer3)
		self.fc2 = nn.Linear(layer3, 1)

	def forward(self, x):

		batchsize = x.shape[0]

		h = torch.eye(node).expand(batchsize,node,node).cuda()

		h = self.gc1(x, h)
		h = F.relu(h)
		x, h = self.pool1(x, h)
		h = self.gc2(x, h)
		h = F.relu(h)
		x, h = self.pool2(x, h)
		h = torch.mean(h, 1)
		h = self.fc1(h)
		h = F.relu(h)
		h = self.fc2(h)

		return h

class Ra_GC(nn.Module):

	def __init__(self, node, cluster, in_dim, out_dim):
		super(Ra_GC, self).__init__()
		self.in_dim = in_dim
		self.out_dim = out_dim
		self.alpha = nn.Parameter(torch.Tensor(node, cluster))
		self.beta = nn.Parameter(torch.Tensor(cluster, in_dim * out_dim))
		self.bias = nn.Parameter(torch.Tensor(out_dim))
		nn.init.xavier_uniform_(self.alpha)
		nn.init.xavier_uniform_(self.beta)
		nn.init.constant_(self.bias, 0)

	def forward(self, a, h):
		w = self.alpha @ self.beta
		w = w.reshape(-1, self.in_dim, self.out_dim)
		h = h.unsqueeze(2)
		h = h @ w + self.bias
		h = h.squeeze(2)
		h = a @ h + h
		return h

class GAT(nn.Module):

	def __init__(self, in_dim, out_dim):
		super(GAT, self).__init__()
		self.proj = nn.Linear(in_dim, out_dim)
		self.attention = nn.Linear(out_dim * 2, 1)
		self.out_dim = out_dim

	def forward(self, a, h):
		batch = h.shape[0]
		node = h.shape[1]
		h = self.proj(h)
		h1 = h.unsqueeze(2).expand(batch,node,node,self.out_dim)
		h2 = h1.permute(0,2,1,3)
		h1 = torch.cat((h1, h2), -1)
		h1 = self.attention(h1)
		h1 = F.relu(h1.squeeze(-1))
		h1 = torch.exp(h1) * a
		h1 = h1 / torch.sum(h1,dim=-1,keepdim=True)
		h = h1 @ h + h
		return h

class Pool(nn.Module):

	def __init__(self, k, in_dim):
		super(Pool, self).__init__()
		self.k = k
		self.sigmoid = nn.Sigmoid()
		self.proj = nn.Linear(in_dim, 1)

	def forward(self, adj, hidden):
		weights = self.proj(hidden).squeeze()
		scores = self.sigmoid(weights)

		batch = adj.shape[0]
		num_nodes = adj.shape[1]
		down_nodes = max(2, int(self.k * num_nodes))
		values, idx = torch.topk(scores, down_nodes)
		print(values, idx)
		batch_idx = torch.tensor(range(batch)).unsqueeze(-1).expand(batch, down_nodes).reshape(-1)
		# batch_idx = batch_idx.cuda()
		values_idx = idx.reshape(-1)
		hidden = hidden[batch_idx, values_idx, :].reshape(batch, down_nodes, -1)
		values = torch.unsqueeze(values, -1)
		hidden = torch.mul(hidden, values)
		adj = adj[batch_idx, values_idx, :].reshape(batch, down_nodes, -1)
		adj = adj[batch_idx, :, values_idx].reshape(batch, -1, down_nodes)
		degrees = torch.sum(adj, -1, keepdim=True)
		adj = adj / degrees

		return adj, hidden

class PR_GNN_large(nn.Module):
	def __init__(self, layer1, layer2, layer3, layer4):
		super(PR_GNN_large, self).__init__()

		print('Current model: PR_GNN_large')

		self.gat1 = GAT(node, layer1)
		self.pool1 = Pool(0.5, layer1)
		self.gat2 = GAT(layer1, layer2)
		self.pool2 = Pool(0.5, layer2)
		self.gat3 = GAT(layer2, layer3)
		self.fc1 = nn.Linear(layer3, layer4)
		self.fc2 = nn.Linear(layer4, 1)

	def forward(self, x):

		batchsize = x.shape[0]

		h = torch.eye(node).expand(batchsize,node,node).cuda()

		h = self.gat1(x, h)
		h = F.relu(h)
		x, h = self.pool1(x, h)
		h = self.gat2(x, h)
		h = F.relu(h)
		x, h = self.pool2(x, h)
		h = self.gat3(x, h)
		h = F.relu(h)
		h = torch.mean(h, 1)
		h = self.fc1(h)
		h = F.relu(h)
		h = self.fc2(h)

		return h

class GPC(nn.Module):

	def __init__(self, in_dim, out_dim):
		super(GPC, self).__init__()
		self.out_dim = out_dim
		self.conv = nn.Conv2d(in_dim, out_dim, (1,node))
		nn.init.normal_(self.conv.weight, std=math.sqrt(2*(1-lamda)/(node*in_dim+node*out_dim)))

	def forward(self, x):
		batchsize = x.shape[0]

		x_c = self.conv(x)
		x_C = x_c.expand(batchsize,self.out_dim,node,node)
		x_R = x_C.permute(0,1,3,2)
		x = x_C+x_R

		return x

class GPC_Res(nn.Module):

	def __init__(self, in_dim, out_dim):
		super(GPC_Res, self).__init__()
		self.out_dim = out_dim
		self.conv = nn.Conv2d(in_dim, out_dim, (1,node))
		nn.init.normal_(self.conv.weight, std=math.sqrt(2*(1-lamda)/(node*in_dim+node*out_dim)))
		self.convres = nn.Conv2d(in_dim, out_dim, 1)
		nn.init.normal_(self.convres.weight, std=math.sqrt(4*lamda/(in_dim+out_dim)))

	def forward(self, x):
		batchsize = x.shape[0]

		res = self.convres(x)
		x_c = self.conv(x)
		x_C = x_c.expand(batchsize,self.out_dim,node,node)
		x_R = x_C.permute(0,1,3,2)
		x = x_C+x_R+res

		return x

class GPC_SE(nn.Module):

	def __init__(self, in_dim, out_dim):
		super(GPC_SE, self).__init__()
		self.out_dim = out_dim
		self.conv = nn.Conv2d(in_dim, out_dim, (1,node))
		nn.init.normal_(self.conv.weight, std=math.sqrt(2*(1-lamda)/(node*in_dim+node*out_dim)))
		self.convres = nn.Conv2d(in_dim, out_dim, 1)
		nn.init.normal_(self.convres.weight, std=math.sqrt(4*lamda/(in_dim+out_dim)))

		self.sed = nn.Linear(out_dim, int(out_dim/SE), False)
		self.seu = nn.Linear(int(out_dim/SE), out_dim, False)

	def forward(self, x):
		batchsize = x.shape[0]

		res = self.convres(x)
		x_c = self.conv(x)
		x_C = x_c.expand(batchsize,self.out_dim,node,node)
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
		self.conv = nn.Conv2d(in_dim, out_dim, (1,node))
		nn.init.normal_(self.conv.weight, std=math.sqrt(4/(node*in_dim+out_dim)))


	def forward(self, x):

		x = self.conv(x)

		return x

class NP(nn.Module):

	def __init__(self, in_dim, out_dim):
		super(NP, self).__init__()
		self.conv = nn.Conv2d(in_dim, out_dim, (node,1))
		nn.init.normal_(self.conv.weight, std=math.sqrt(4/(node*in_dim+out_dim)))

	def forward(self, x):

		x = self.conv(x)

		return x

class BC_GCN(nn.Module):
	def __init__(self, GPC_dim_1, GPC_dim_2, GPC_dim_3, EP_dim, NP_dim):
		super(BC_GCN, self).__init__()

		print('Current model: BC_GCN')

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

		print('Current model: BC_GCN_Res')

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

		print('当前模型:BC_GCN_SE')

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

if __name__ == '__main__':
	pool = Pool(0.5, 128)
	a = torch.rand(64, 112, 112)
	h = torch.rand(64, 112, 128)
	pool(a, h)
