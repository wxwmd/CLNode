import copy
import sys

import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid

sys.path.append("..")
from util import random_split, training_scheduler, sort
from GCN import GCNNet
from early_stop import EarlyStop
import argparse

parser = argparse.ArgumentParser(description="progrom description")
parser.add_argument('--percent', default=2)
parser.add_argument('--lam', default=0.25)
parser.add_argument('--T', default=50)
parser.add_argument('--scheduler', default='geom')
args = parser.parse_args()

percent = float(args.percent) / 100
lam = float(args.lam)
T = float(args.T)
scheduler = args.scheduler

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset = Planetoid(root='./data/Cora', name='Cora')
data = dataset[0].to(device)

# split data
random_split(data, percent)

model1 = GCNNet(dataset, 16, dataset.num_classes).to(device)
model2 = GCNNet(dataset, 16, dataset.num_classes).to(device)


# ---------------------measure difficulty with f1----------------------------
patience = 50
early_stop1 = EarlyStop(patience, './best_model.pth')
optimizer1 = torch.optim.Adam(model1.parameters(), lr=0.01, weight_decay=5e-4)

for epoch in range(500):
    model1.train()
    optimizer1.zero_grad()
    out, _ = model1(data)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])

    loss.backward()
    optimizer1.step()

    model1.eval()
    out, _ = model1(data)
    _, pred = out.max(dim=1)
    correct = int(pred[data.val_mask].eq(data.y[data.val_mask]).sum().item())
    acc = correct / int(data.val_mask.sum())
    #print(f'epoch: {epoch}, acc:{acc}')

    if not early_stop1.step(acc, model1):
        pre_model = torch.load('./best_model.pth')
        break

# 测试
model1.eval()
out, embedding = model1(data)
_, pred = out.max(dim=1)

label = copy.deepcopy(pred)
label[data.train_mask] = data.y[data.train_mask]
# 将训练集按照难度排序
sorted_trainset = sort(data, label)

# ------------ train with clnode--------------
early_stop2 = EarlyStop(patience, './best_model.pth')
optimizer2 = torch.optim.Adam(model2.parameters(), lr=0.01, weight_decay=5e-4)

for epoch in range(1, 500):
    size = training_scheduler(lam, epoch, T, scheduler)
    training_subset = sorted_trainset[:int(size * sorted_trainset.shape[0])]
    optimizer2.zero_grad()
    model2.train()
    out,_ = model2(data)
    loss = F.nll_loss(out[training_subset], data.y[training_subset])

    loss.backward()
    optimizer2.step()

    # 在验证集上计算准确率
    model2.eval()
    out,_ = model2(data)
    _, pred = out.max(dim=1)
    correct = int(pred[data.val_mask].eq(data.y[data.val_mask]).sum().item())
    acc = correct / int(data.val_mask.sum())
    #print(f'epoch: {epoch}, acc:{acc}')

    # early stop
    if not early_stop2.step(acc, model2):
        model2 = torch.load('./best_model.pth')
        break

# 测试
model2.eval()
out,_ = model2(data)
_, pred = out.max(dim=1)
correct = int(pred[data.test_mask].eq(data.y[data.test_mask]).sum().item())
acc = correct / int(data.test_mask.sum())
print('{:.4f}'.format(acc), end='\t')
