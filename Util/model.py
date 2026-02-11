import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

class GNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout, num_properties):
        super(GNNModel, self).__init__()
        
        # Path 1: Graph Network (GCN + FC)
        self.graph_convs = nn.ModuleList()
        self.graph_convs.append(GCNConv(input_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.graph_convs.append(GCNConv(hidden_dim, hidden_dim))
        self.graph_fc = nn.Linear(hidden_dim, 8)  # Output: 8 features
        
        # Path 2: Input1 Network (Element Properties)
        self.input1_fc1 = nn.Linear(num_properties, hidden_dim)
        self.input1_fc2 = nn.Linear(hidden_dim, 8)  # Output: 8 features
        
        # Path 3: Input2 Network (Coordination Number + Enthalpy)
        self.input2_fc1 = nn.Linear(2, hidden_dim)
        self.input2_fc2 = nn.Linear(hidden_dim, 8)  # Output: 8 features
        
        # Final Network (24 -> 12 -> 1)
        self.final_fc1 = nn.Linear(24, 12)  # 24 = 8 + 8 + 8
        self.final_fc2 = nn.Linear(12, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, data):
        batch_size = data.num_graphs
        
        # Path 1: Graph Network
        x, edge_index = data.x, data.edge_index
        graph_x = x
        for conv in self.graph_convs:
            graph_x = conv(graph_x, edge_index)
            graph_x = self.relu(graph_x)
            graph_x = self.dropout(graph_x)
        graph_x = global_mean_pool(graph_x, data.batch)  # Shape: [batch_size, hidden_dim]
        graph_x = self.graph_fc(graph_x)  # Shape: [batch_size, 8]
        graph_x = self.relu(graph_x)
        
        # Path 2: Input1 Network
        data_list = data.to_data_list()
        input1_list = []
        for d in data_list:
            element_props = d.input1  # Shape: [num_non_I_elements, num_properties]
            element_features = self.input1_fc1(element_props)
            element_features = self.relu(element_features)
            element_features = self.dropout(element_features)
            element_features = self.input1_fc2(element_features)
            element_features = self.relu(element_features)
            structure_features = torch.mean(element_features, dim=0)
            input1_list.append(structure_features)
        input1_x = torch.stack(input1_list)
        
        # Path 3: Input2 Network
        input2_list = []
        for d in data_list:
            input2_list.append(d.input2)
        input2_x = torch.stack(input2_list)
        input2_x = self.input2_fc1(input2_x)
        input2_x = self.relu(input2_x)
        input2_x = self.dropout(input2_x)
        input2_x = self.input2_fc2(input2_x)
        input2_x = self.relu(input2_x)
        
        # Combine all paths
        combined_x = torch.cat([graph_x, input1_x, input2_x], dim=1)
        
        # Final network
        x = self.final_fc1(combined_x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.final_fc2(x)
        
        return x


