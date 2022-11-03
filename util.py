import math

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
def difficulty_measurer(data, label, embedding, alpha):
    local_difficulty = neighborhood_difficulty_measurer(data, label)
    global_difficulty = feature_difficulty_measurer(data, label, embedding)
    node_difficulty = alpha * local_difficulty + (1 - alpha) * global_difficulty
    return node_difficulty


# sort training nodes by difficulty
def sort_training_nodes(data, label, embedding, alpha = 0.5):
    node_difficulty = difficulty_measurer(data, label, embedding, alpha)
    _, indices = torch.sort(node_difficulty)
    sorted_trainset = data.train_id[indices]
    return sorted_trainset


def training_scheduler(lam, t, T, scheduler='linear'):
    if scheduler == 'linear':
        return min(1, lam + (1 - lam) * t / T)
    elif scheduler == 'root':
        return min(1, math.sqrt(lam ** 2 + (1 - lam ** 2) * t / T))
    elif scheduler == 'geom':
        return min(1, 2 ** (math.log2(lam) - math.log2(lam) * t / T))