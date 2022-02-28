import torch
import torch.optim as optim
import torch.nn as nn
import time
import numpy as np
import math

import model
import config

torch.backends.cudnn.benchmark = True

def setup_seed(seed):
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)

setup_seed(1)

cuda = config.cuda
print('cuda:',cuda)
if cuda != '-1':
	import os
	os.environ["CUDA_VISIBLE_DEVICES"] = cuda

scale = config.scale

train_data = torch.rand(90, 10, 112, 112)  # subjects * slices * node * node
train_data = train_data.view(-1, 112, 112)
train_data = train_data[torch.randperm(train_data.size(0))]  # shuffle train data
train_data = train_data.view(15, 60, 1, 112, 112)  # batch number * batch size * 1 * node * node
train_label = (torch.rand(90) * 800).int()  # labels range from 0 to 800 days
train_label = train_label.unsqueeze(1).expand(90, 10).reshape(15, 60)
train_label = train_label / scale

test_data = torch.rand(10, 10, 112, 112)  # subjects * slices * node * node
test_data = test_data.unsqueeze(2)
test_label = (torch.rand(10) * 800).int()  # labels range from 0 to 800 days
test_label = test_label.unsqueeze(1).expand(10, 10)
test_label = test_label / scale

criterion = nn.L1Loss()

# net = model.BC_GCN(16, 16, 16, 64, 256)
# net = model.BC_GCN_Res(16, 16, 16, 64, 256)
net = model.BC_GCN_SE(16, 16, 16, 64, 256)

net.apply(model.weights_init)
total = sum([param.nelement() for param in net.parameters()])
print("Number of parameter: %.3fM" % (total/1e6))
if config.cuda != '-1':
	net = nn.DataParallel(net)
	net.cuda()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9, weight_decay=0)

loss_best = 800
loss_s_best = 800
starttime = time.time()
for epoch in range(50):

	net.train()
	running_loss = 0.0
	for i in range(train_label.size(0)):
		inputs = train_data[i]
		labels = train_label[i]
		if config.cuda != '-1':
			inputs=inputs.cuda()
			labels=labels.cuda()
		labels = labels.float().unsqueeze(1)
		optimizer.zero_grad()
		outputs = net(inputs)
		loss = criterion(outputs, labels)
		loss.backward()
		optimizer.step()
		running_loss += loss.item()/train_label.size(0)*scale

	net.eval()
	test_loss = 0.0
	test_loss_s = 0.0
	with torch.no_grad():
		for i in range(test_label.size(0)):
			inputs = test_data[i]
			labels = test_label[i]
			if config.cuda != '-1':
				inputs=inputs.cuda()
				labels=labels.cuda()
			outputs = net(inputs)
			labels = labels.float().unsqueeze(1)
			loss = criterion(outputs, labels)
			test_loss += loss.item()/test_label.size(0)*scale

			output_s = np.mean(outputs.cpu().numpy())
			label_s = labels.cpu().numpy()[0]
			loss = abs(output_s-label_s)[0]
			test_loss_s += loss/test_label.size(0)*scale

	ltime = time.time()-starttime

	if(loss_best>test_loss):
		loss_best = test_loss
	if(loss_s_best>test_loss_s):
		loss_s_best = test_loss_s
		print('Min loss')
		# torch.save(net.state_dict(), './save/' + config.modelname +'.pkl')

	print('[%d]loss:%.3f testloss:%.4f testloss_s:%.4f time:%.2fm' %
	(epoch + 1, running_loss, test_loss, test_loss_s, ltime/60))

	if math.isnan(running_loss):
		print('break')
		break

print('Min loss: %.4f  %.4f'%(loss_best, loss_s_best))
