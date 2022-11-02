import math

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.utils import add_self_loops
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# neighborhood-based difficulty measurer
def neighborhood_difficulty_measurer(data, label):
    # 加上自环，将节点本身的标签也计算在内
    neighbor_label, _ = add_self_loops(data.edge_index)
    # 得到每个节点的邻居标签
    neighbor_label[1] = label[neighbor_label[1]]
    # 得到训练集中每个节点的邻节点分布
    neighbor_label = torch.transpose(neighbor_label, 0, 1)
    index, count = torch.unique(neighbor_label, sorted=True, return_counts=True, dim=0)
    neighbor_class = torch.sparse_coo_tensor(index.T, count)
    neighbor_class = neighbor_class.to_dense().float()
    # 开始计算节点的邻居信息熵

    neighbor_class = neighbor_class[data.train_id]
    neighbor_class = F.normalize(neighbor_class, 1.0, 1)
    neighbor_entropy = -1 * neighbor_class * torch.log(neighbor_class + torch.exp(torch.tensor(-20)))  # 防止log里面是0出现异常
    local_difficulty = neighbor_entropy.sum(1)
    return local_difficulty.to(device)


# feature-based difficulty measurer
def feature_difficulty_measurer(data, label, embedding):
    normalized_embedding = F.normalize(torch.exp(embedding))
    classes = label.unique()
    class_features = {}
    for c in classes:
        class_nodes = torch.nonzero(label == c).squeeze(1)
        node_features = normalized_embedding.index_select(0, class_nodes)
        class_feature = node_features.sum(dim=0)
        # 这里注意归一化
        class_feature = class_feature / torch.sqrt((class_feature * class_feature).sum())
        class_features[c.item()] = class_feature

    similarity = {}
    for u in data.train_id:
        # 做了实验，认为让节点乘错误的类别feature，看看效果
        feature = normalized_embedding[u]
        class_feature = class_features[label[u].item()]
        sim = torch.dot(feature, class_feature)
        sum = torch.tensor(0.).to(device)
        for cf in class_features.values():
            sum += torch.dot(feature, cf)
        sim = sim * len(classes) / sum
        similarity[u.item()] = sim

    class_avg = {}
    for c in classes:
        count = 0.
        sum = 0.
        for u in data.train_id:
            if label[u] == c:
                count += 1
                sum += similarity[u.item()]
        class_avg[c.item()] = sum / count

    global_difficulty = []

    for u in data.train_id:
        sim = similarity[u.item()] / class_avg[label[u].item()]
        # print(u,sim)
        sim = torch.tensor(1) if sim > 0.95 else sim
        node_difficulty = 1 / sim
        global_difficulty.append(node_difficulty)

    return torch.tensor(global_difficulty).to(device)


# multi-perspective difficulty measurer
def difficulty_measurer(data, label, embedding):
    local_difficulty = neighborhood_difficulty_measurer(data, label)
    global_difficulty = feature_difficulty_measurer(data, label, embedding)
    node_difficulty = local_difficulty + global_difficulty
    return node_difficulty


# sort training nodes by difficulty
def sort_training_nodes(data, label, embedding):
    node_difficulty = difficulty_measurer(data, label, embedding)
    _, indices = torch.sort(node_difficulty)
    sorted_trainset = data.train_id[indices]
    return sorted_trainset

def sort(data, label):
    difficulty = neighborhood_difficulty_measurer(data, label)
    _, indices = torch.sort(difficulty)
    sorted_trainset = data.train_id[indices]
    return sorted_trainset


def training_scheduler(lam, t, T, scheduler='linear'):
    if scheduler == 'linear':
        return min(1, lam + (1 - lam) * t / T)
    elif scheduler == 'root':
        return min(1, math.sqrt(lam ** 2 + (1 - lam ** 2) * t / T))
    elif scheduler == 'geom':
        return min(1, 2 ** (math.log2(lam) - math.log2(lam) * t / T))


# 为数据集得到20 nodes per class的训练集
def standard_split(data, classes=5):
    data.train_mask = torch.full(data.y.shape, False)
    data.val_mask = torch.full(data.y.shape, False)
    data.test_mask = torch.full(data.y.shape, False)
    for i in range(0, classes):
        count = 0
        for index in range(10000):
            if data.y[index] == i:
                data.train_mask[index] = True
                count += 1
                if count >= 20:
                    break
    data.val_mask[-1500:-1000] = True
    data.test_mask[-1000:] = True
    data.train_id = data.train_mask.nonzero().squeeze(dim=1)
    return data


def random_split(data, percent):
    node_id = np.arange(data.num_nodes)
    np.random.shuffle(node_id)
    training_nodes_num = int(data.num_nodes * percent)
    train_ids = torch.tensor(node_id[:training_nodes_num], dtype=torch.long)
    val_ids = torch.tensor(node_id[training_nodes_num:training_nodes_num + 500], dtype=torch.long)
    test_ids = torch.tensor(node_id[training_nodes_num + 500:training_nodes_num + 1500], dtype=torch.long)

    data.train_mask = torch.full(data.y.shape, False)
    data.train_mask[train_ids] = True
    data.val_mask = torch.full(data.y.shape, False)
    data.val_mask[val_ids] = True
    data.test_mask = torch.full(data.y.shape, False)
    data.test_mask[test_ids] = True
    data.train_id = data.train_mask.nonzero().squeeze(dim=1)
    return data


# ------------------------------------------------
# 消融实验使用的difficulty measurer
def sort_training_nodes_local(data, label, embedding):
    node_difficulty = neighborhood_difficulty_measurer(data, label)
    _, indices = torch.sort(node_difficulty)
    sorted_trainset = data.train_id[indices]
    return sorted_trainset


def sort_training_nodes_global(data, label, embedding):
    node_difficulty = feature_difficulty_measurer(data, label, embedding)
    _, indices = torch.sort(node_difficulty)
    sorted_trainset = data.train_id[indices]
    return sorted_trainset
