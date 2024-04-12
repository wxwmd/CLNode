import copy
import sys

import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid

sys.path.append("..")
from util import training_scheduler, sort_training_nodes, setup_seed, get_noisy_data
from GCN import GCNNet, GCNClassifier
from early_stop import EarlyStop
from setting import device

import argparse

setup_seed(0)

parser = argparse.ArgumentParser(description="program description")
parser.add_argument('--percent', default=1)
parser.add_argument('--scheduler', default='geom')
args = parser.parse_args()

percent = float(args.percent) / 100
scheduler = args.scheduler

NUM_EPOCHS = 500
PATIENCE = 50

dataset = Planetoid(root='./data/Cora', name='Cora')
data = dataset[0].to(device)

data.num_classes = 7

data = get_noisy_data(data, 0.1)


# ---------------------CLNode------------------------------

# ---------------------CLNode中的f1-------------------------
pre_model = GCNClassifier(dataset, 16, 16).to(device)
pre_early_stop = EarlyStop(PATIENCE, './checkpoints/best_pre_model.pth')
pre_optimizer = torch.optim.Adam(pre_model.parameters(), lr=0.01, weight_decay=5e-4)

for epoch in range(NUM_EPOCHS):
    pre_model.train()
    pre_optimizer.zero_grad()
    _, out = pre_model(data)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])

    loss.backward()
    pre_optimizer.step()

    pre_model.eval()
    _, out = pre_model(data)
    _, pred = out.max(dim=1)
    correct = int(pred[data.val_mask].eq(data.y[data.val_mask]).sum().item())
    acc = correct / int(data.val_mask.sum())

    if not pre_early_stop.step(acc, pre_model):
        break

# 测试
pre_model = torch.load('./checkpoints/best_pre_model.pth')
pre_model.eval()
embedding, out = pre_model(data)
_, pred = out.max(dim=1)

label = copy.deepcopy(pred)
label[data.train_mask] = data.y[data.train_mask]
# 将训练集按照难度排序
sorted_trainset = sort_training_nodes(data, label, embedding, alpha=1)

# ------------------- CLNode中的f2，真正用于预测的那个----------------

# 网格搜索最优的lambda和T
best_lambda = 0
best_T = 0
best_val_acc = 0
for lam in [0.25, 0.5, 0.75]:
    for T in [50, 100, 200]:
        model_gs = GCNNet(dataset, 16, dataset.num_classes).to(device)
        optimizer_gs = torch.optim.Adam(model_gs.parameters(), lr=0.01, weight_decay=5e-4)
        early_stop_gs = EarlyStop(PATIENCE, './checkpoints/best_model-' + str(lam) + '-' + str(T) + '.pth')
        for epoch in range(NUM_EPOCHS):
            size = training_scheduler(lam, epoch, T, scheduler)
            batch_id = sorted_trainset[:int(size * sorted_trainset.shape[0])]
            optimizer_gs.zero_grad()
            model_gs.train()
            out = model_gs(data)
            loss = F.nll_loss(out[batch_id], data.y[batch_id])

            loss.backward()
            optimizer_gs.step()

            # 在验证集上计算准确率
            model_gs.eval()
            _, pred = model_gs(data).max(dim=1)
            correct = int(pred[data.val_mask].eq(data.y[data.val_mask]).sum().item())
            acc = correct / int(data.val_mask.sum())

            # early stop
            if not early_stop_gs.step(acc, model_gs):
                break

        model_gs = torch.load('./checkpoints/best_model-' + str(lam) + '-' + str(T) + '.pth')
        model_gs.eval()
        _, pred = model_gs(data).max(dim=1)
        correct = int(pred[data.val_mask].eq(data.y[data.val_mask]).sum().item())
        val_acc = correct / int(data.val_mask.sum())
        print('the lambda is {:.2f}, the T is {}, the val_acc is {:.4f}'.format(lam, T, val_acc), end='\n')
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_lambda = lam
            best_T = T



# 使用best_lambda和best_T训练
model_clnode = GCNNet(dataset, 16, dataset.num_classes).to(device)
optimizer_clnode = torch.optim.Adam(model_clnode.parameters(), lr=0.01, weight_decay=5e-4)
early_stop_clnode = EarlyStop(PATIENCE, './checkpoints/best_model-clnode.pth')
for epoch in range(NUM_EPOCHS):
    size = training_scheduler(best_lambda, epoch, best_T, scheduler)
    batch_id = sorted_trainset[:int(size * sorted_trainset.shape[0])]
    optimizer_clnode.zero_grad()
    model_clnode.train()
    out = model_clnode(data)
    loss = F.nll_loss(out[batch_id], data.y[batch_id])

    loss.backward()
    optimizer_clnode.step()

    # 在验证集上计算准确率
    model_clnode.eval()
    _, pred = model_clnode(data).max(dim=1)
    correct = int(pred[data.val_mask].eq(data.y[data.val_mask]).sum().item())
    acc = correct / int(data.val_mask.sum())

    # early stop
    if not early_stop_clnode.step(acc, model_clnode):
        break


model = torch.load('./checkpoints/best_model-clnode.pth')
model.eval()
_, pred = model(data).max(dim=1)
correct = int(pred[data.test_mask].eq(data.y[data.test_mask]).sum().item())
acc_clnode = correct / int(data.test_mask.sum())
print('the best lambda is {:.2f}, the best T is {}'.format(best_lambda, best_T), end='\n')
print('the accuracy of CLNode is {:.4f}'.format(acc_clnode), end='\n')

#-------------------------------------------------------------------
# train a backbone model to compare the accuracy
backbone_model = GCNClassifier(dataset, 16, 16).to(device)
backbone_early_stop = EarlyStop(PATIENCE, './checkpoints/best_backbone_model.pth')
backbone_optimizer = torch.optim.Adam(backbone_model.parameters(), lr=0.01, weight_decay=5e-4)

for epoch in range(NUM_EPOCHS):
    backbone_model.train()
    backbone_optimizer.zero_grad()
    _, out = backbone_model(data)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])

    loss.backward()
    backbone_optimizer.step()

    backbone_model.eval()
    _, out = backbone_model(data)
    _, pred = out.max(dim=1)
    correct = int(pred[data.val_mask].eq(data.y[data.val_mask]).sum().item())
    acc = correct / int(data.val_mask.sum())

    if not backbone_early_stop.step(acc, backbone_model):
        break

backbone_model = torch.load('./checkpoints/best_backbone_model.pth')
backbone_model.eval()
embedding, out = backbone_model(data)
_, pred = out.max(dim=1)
correct = int(pred[data.test_mask].eq(data.y[data.test_mask]).sum().item())
acc_backbone = correct / int(data.test_mask.sum())

print('the accuracy of backbone is {:.4f}'.format(acc_backbone), end='\n')
print('the improvement of clnode is {:.1f}%'.format(100*(acc_clnode - acc_backbone)), end='\n')