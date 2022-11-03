import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_sparse import SparseTensor
from torch.nn import Linear

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class GCNNet1(torch.nn.Module):
    def __init__(self, dataset, hidden_num, embedding_num, out_num):
        super(GCNNet1, self).__init__()
        self.conv1 = GCNConv(dataset.num_node_features, hidden_num)
        self.conv2 = GCNConv(hidden_num, embedding_num)
        self.linear = Linear(embedding_num, out_num)

    def forward(self, data):
        x, edge_index = data.x, SparseTensor.from_edge_index(data.edge_index).to(device)
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.dropout(x, training=self.training)
        embedding = x.data
        x = self.linear(x)
        x = F.log_softmax(x, dim=1)
        return x, embedding

class GCNNet2(torch.nn.Module):
    def __init__(self, dataset, hidden_num, out_num):
        super(GCNNet2, self).__init__()
        self.conv1 = GCNConv(dataset.num_node_features, hidden_num)
        self.conv2 = GCNConv(hidden_num, out_num)

    def forward(self, data):
        x, edge_index = data.x, SparseTensor.from_edge_index(data.edge_index).to(device)
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.dropout(x, training=self.training)
        return F.log_softmax(x, dim=1)


