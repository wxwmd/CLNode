import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_sparse import SparseTensor

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class GCNNet(torch.nn.Module):
    def __init__(self, dataset, hidden_num, out_num):
        super(GCNNet, self).__init__()
        self.conv1 = GCNConv(dataset.num_node_features, hidden_num)
        self.conv2 = GCNConv(hidden_num, out_num)

    def forward(self, data):
        x, edge_index = data.x, SparseTensor.from_edge_index(data.edge_index).to(device)
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        embedding = x.data
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.dropout(x, training=self.training)
        return F.log_softmax(x, dim=1), embedding
