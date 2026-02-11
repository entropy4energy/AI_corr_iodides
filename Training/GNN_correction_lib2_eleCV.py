# -*- coding: utf-8 -*-
"""
Created by: Guangshuai (Jerry) Han @S4E Lab
Year: 2025
Affiliation: Johns Hopkins University
Purpose: Using a sample 2 GCN layer model to test the graph, predict "enthalpy_formation_cce_300K_cell" correction
element-wise CV, use one cation as validation set, and the rest as training set.
"""

#=========================================================================
#+++++++++++++++++++++++++++++++++config++++++++++++++++++++++++++++++++++
#=========================================================================
import os
import json
import torch
import sys
training_info = {
    'target': 'enthalpy_formation_cce_300K_atom',
    'add_info': 'enthalpy_formation_atom',
    'struct_type': 'Relaxed Structure',
    'use_diff_target': True,
    'prop_list': [
        "atomic_mass",
        "atomic_number",
        "c6_gb",
        "density_PT",
        "electron_affinity_PT",
        "electronegativity_Allen",
        "electronegativity_Ghosh",
        "enthalpy_vaporization",
        "polarizability",
        "radii_Ghosh08",
        "radii_Pyykko",
        "radii_Slatter",
        "radius_PT",
        "radius_covalent_PT"
    ],
    'xlable': 'True Values',
    'ylable': 'Predictions',
    'title': 'GNN Predictions vs True Values',
    'normalize': 'quantile',
    'graph_cutoff': 5.0,
    'fractional': True,
    'global_attr': 1,
    'json_data_path': 'Data/CCE_data.json',
    'validation_split': 0.2,
    'random_seed': 42,
    'num_layers': 2,
    'hidden_dim': 64,
    'dropout': 0.1,
    'learning_rate': 0.002,
    'num_epochs': 5,
    'batch_size': 32,
    'weight_decay': 5e-4,
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    'test_elements': ['Zn'],
    'exclude_elements': ['Fe','Cs','Sn','Zn','Hg','Ni'],
}
import torch.nn as nn
import torch.optim as optim
import torch_geometric
import torch_geometric.transforms as T
from torch_geometric.data import DataLoader
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import datetime
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../'))

if project_root not in sys.path:
    sys.path.append(project_root)

# Create output folder
output_root = os.path.join(project_root, "Output/Training")
today_str = datetime.datetime.now().strftime("%Y-%m-%d")
time_str = datetime.datetime.now().strftime("%H-%M-%S")
output_dir = os.path.join(output_root, today_str, time_str)

os.makedirs(output_dir, exist_ok=True)  # Ensure directories exist

# Import necessary functions
from Util.graph_loader import (
    ele_df,
    StructureGraphBuilder
)
from Util.visual_func import plot_results
from Util.model import GNNModel
# Load element data
ele_df = ele_df(prop_list=training_info['prop_list'], normalize=training_info['normalize'])
# Load JSON data
print(ele_df)
json_file = training_info['json_data_path']
# Read the JSON file
with open(json_file, 'r') as f:
    data = json.load(f)


print("Data loaded successfully.")
import torch
import numpy as np

def check_for_nan(data_loader, model, device):
    """
    Check whether any NaNs exist in target or model predictions.
    """
    model.eval()
    found_nan = False
    for i, batch in enumerate(data_loader):
        batch = batch.to(device)
        y_true = batch.y.view(-1).cpu().numpy()

        # Check target
        if np.any(np.isnan(y_true)):
            print(f"❗ NaN detected in target values at batch {i}")
            found_nan = True

        with torch.no_grad():
            y_pred = model(batch).view(-1).cpu().numpy()

        # Check prediction
        if np.any(np.isnan(y_pred)):
            print(f"❗ NaN detected in model predictions at batch {i}")
            found_nan = True

    if not found_nan:
        print("✅ No NaN values found in targets or predictions.")

# Convert JSON data to graph data
builder = StructureGraphBuilder(ele_df, cutoff=training_info['graph_cutoff'], fractional=training_info['fractional'])

train_graph_data_list = []
train_target_output = []
test_graph_data_list = []
test_target_output = []

add_info_list_train, add_info_list_test = [], []
element_list_train, element_list_test = [], []
sg_list_train, sg_list_test = [], []
cn_list_train, cn_list_test = [], []  # Store coordination numbers
enthalpy_list_train, enthalpy_list_test = [], []  # Store enthalpy formation

for i in range(len(data)):
    temp_ele = list(set([x for x in data[i]['Structure']['Initial Structure']['Elements'] if x != "I"]))
    temp_target = data[i]['Main'].get(training_info['target'])
    temp_add_info = data[i]['Main'].get(training_info['add_info'])
    temp_sg = data[i]['Main'].get('sg', None)  # Extract space group information
    temp_graph = builder.build_graph(data[i]['Structure'][training_info['struct_type']], global_attr=training_info['global_attr'])

    # Get average coordination number from JSON
    avg_cn = data[i].get('average_coordination_number')
    
    # Get enthalpy formation
    enthalpy = data[i]['Main'].get('enthalpy_formation_atom')

    target_raw = data[i]['Main'].get(training_info['target'])
    add_info_raw = data[i]['Main'].get(training_info['add_info'])

    if training_info.get("use_diff_target", False):
        if target_raw is None or add_info_raw is None or enthalpy is None or avg_cn is None:
            continue
        temp_target = target_raw - add_info_raw
        temp_add_info = add_info_raw
    else:
        if target_raw is None or enthalpy is None or avg_cn is None:
            continue
        temp_target = target_raw
        temp_add_info = add_info_raw

    # Skip samples with no non-I elements
    if len(temp_ele) == 0:
        continue

    # 使用测试元素和排除元素逻辑
    # 如果样本包含任何测试元素，则归入测试集
    if any(x in temp_ele for x in training_info['test_elements']):
        test_graph_data_list.append(temp_graph)
        test_target_output.append(temp_target)
        add_info_list_test.append(temp_add_info)
        element_list_test.append(temp_ele)
        sg_list_test.append(temp_sg)  # Save space group for test set
        cn_list_test.append(avg_cn)  # Store average coordination number
        enthalpy_list_test.append(enthalpy)
    # 如果样本不包含任何要排除的元素，则归入训练集
    elif not any(x in temp_ele for x in training_info['exclude_elements']):
        train_graph_data_list.append(temp_graph)
        train_target_output.append(temp_target)
        add_info_list_train.append(temp_add_info)
        element_list_train.append(temp_ele)
        sg_list_train.append(temp_sg)  # Save space group for training set
        cn_list_train.append(avg_cn)  # Store average coordination number
        enthalpy_list_train.append(enthalpy)

print("train graph data list length:", len(train_graph_data_list))
print("test graph data list length:", len(test_graph_data_list))
#build the dataset and dataloader
from torch_geometric.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, graph_data_list, target_output, add_info_list, element_list, cn_list, enthalpy_list):
        super().__init__()
        self.graph_data_list = graph_data_list
        self.target_output = target_output
        self.add_info_list = add_info_list
        self.element_list = element_list
        self.cn_list = cn_list
        self.enthalpy_list = enthalpy_list
        self.num_properties = len(training_info['prop_list'])

    def __len__(self):
        return len(self.graph_data_list)

    def __getitem__(self, idx):
        data = self.graph_data_list[idx]
        data.y = torch.tensor([self.target_output[idx]], dtype=torch.float)
        data.add_info = torch.tensor([self.add_info_list[idx]], dtype=torch.float)
        data.element = self.element_list[idx]  # For CSV logging only
        
        # Process element properties (input1)
        # Filter out I element and get properties of remaining elements
        elements = [x for x in self.element_list[idx] if x != "I"]
        # Get properties for non-I elements
        element_props = ele_df.loc[:, elements].T.values  # Shape: [num_non_I_elements, num_properties]
        data.input1 = torch.tensor(element_props, dtype=torch.float)  # Shape: [num_non_I_elements, num_properties]
        
        # Process coordination number and enthalpy (input2)
        avg_cn = self.cn_list[idx]
        data.input2 = torch.tensor([avg_cn, self.enthalpy_list[idx]], dtype=torch.float)  # Shape: [2]
        
        return data
    
# Create dataset and dataloader
train_dataset = CustomDataset(train_graph_data_list, train_target_output, add_info_list_train, element_list_train, cn_list_train, enthalpy_list_train)
val_dataset = CustomDataset(test_graph_data_list, test_target_output, add_info_list_test, element_list_test, cn_list_test, enthalpy_list_test)
train_loader = DataLoader(train_dataset, batch_size=training_info['batch_size'], shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=training_info['batch_size'], shuffle=False)

#print train and val size
print("train size:", len(train_loader.dataset), "val size:", len(val_loader.dataset))
# Initialize the model, loss function, and optimizer
input_dim = train_graph_data_list[0].x.shape[1] 
output_dim = 1

model = GNNModel(
    input_dim=input_dim,
    hidden_dim=training_info['hidden_dim'],
    output_dim=output_dim,
    num_layers=training_info['num_layers'],
    dropout=training_info['dropout'],
    num_properties=len(training_info['prop_list'])
).to(training_info['device'])
print("graph_data_list[0].x.shape =", train_graph_data_list[0].x.shape)
check_for_nan(train_loader, model, training_info["device"])
check_for_nan(val_loader, model, training_info["device"])

criterion = nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=training_info['learning_rate'], weight_decay=training_info['weight_decay'])
# Training loop
num_epochs = training_info['num_epochs']
train_loss_list = []
val_loss_list = []

#save the model with the best validation loss
best_val_loss = float('inf')
best_model_path = os.path.join(output_dir, "best_model.pth")
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    for batch in train_loader:
        batch = batch.to(training_info['device'])
        optimizer.zero_grad()
        outputs = model(batch)
        loss = criterion(outputs, batch.y.view(-1, 1).to(training_info['device']))
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(train_loader)
    train_loss_list.append(train_loss)
    print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}")



# Validation
model.eval()
val_loss = 0.0
predictions = []
true_values = []
with torch.no_grad():
    for batch in val_loader:
        batch = batch.to(training_info['device'])
        outputs = model(batch)
        loss = criterion(outputs, batch.y.view(-1, 1).to(training_info['device']))
        val_loss += loss.item()
        predictions.extend(outputs.cpu().numpy().flatten()) 
        true_values.extend(batch.y.cpu().numpy().flatten()) 


#predict on trainig set
train_predictions = []
train_true_values = []
with torch.no_grad():
    for batch in train_loader:
        batch = batch.to(training_info['device'])
        outputs = model(batch)
        train_predictions.extend(outputs.cpu().numpy().flatten()) 
        train_true_values.extend(batch.y.cpu().numpy().flatten()) 
#save the training predictions vs true values to a csv file
import pandas as pd
# Convert sg list to string format for CSV (handle None and list cases)
def format_sg(sg):
    if sg is None:
        return None
    elif isinstance(sg, list):
        return "; ".join(str(s) for s in sg)  # Join multiple space groups with semicolon
    else:
        return str(sg)

train_df = pd.DataFrame({
    'True Values': train_true_values,
    'Predictions': train_predictions,
    'Additional Info': add_info_list_train,
    'Space Group (sg)': [format_sg(sg) for sg in sg_list_train],
    'Elements': ["-".join(ele) for ele in element_list_train]
})

val_df = pd.DataFrame({
    'True Values': true_values,
    'Predictions': predictions,
    'Additional Info': add_info_list_test,
    'Space Group (sg)': [format_sg(sg) for sg in sg_list_test],
    'Elements': ["-".join(ele) for ele in element_list_test]
})

train_df.to_csv(os.path.join(output_dir, "train_predictions.csv"), index=False)
val_df.to_csv(os.path.join(output_dir, "val_predictions.csv"), index=False)

#visualize the training predictions vs true values
plot_results(
    true_values=train_true_values,
    predicted_values=train_predictions,
    xlabel=training_info['xlable'],
    ylabel=training_info['ylable'],
    title=training_info['title'],
    file_path=os.path.join(output_dir, "train_predictions.png")
)
#visualize the validation predictions vs true values
plot_results(
    true_values=true_values,
    predicted_values=predictions,
    xlabel=training_info['xlable'],
    ylabel=training_info['ylable'],
    title=training_info['title'],
    file_path=os.path.join(output_dir, "val_predictions.png")
)
# Save the model
torch.save(model.state_dict(), best_model_path)
print(f"Model saved to {best_model_path}")
# Save training and validation loss
#save the training_info as a json file
import json
training_info_path = os.path.join(output_dir, "training_info.json")
def make_serializable(obj):
    if isinstance(obj, (torch.device, torch.dtype)):
        return str(obj)
    elif isinstance(obj, set):
        return list(obj)
    else:
        return obj

serializable_info = {k: make_serializable(v) for k, v in training_info.items()}

# Add detailed per-sample data with space group and coordination number information to the output JSON
# Create detailed data for each sample in training and validation sets
training_samples = []
for i in range(len(train_true_values)):
    avg_cn = cn_list_train[i]
    training_samples.append({
        'index': i,
        'true_value': float(train_true_values[i]),
        'prediction': float(train_predictions[i]),
        'additional_info': float(add_info_list_train[i]) if add_info_list_train[i] is not None else None,
        'space_group': format_sg(sg_list_train[i]),
        'avg_coordination_number': float(avg_cn) if avg_cn is not None else None,
        'enthalpy_formation_atom': float(enthalpy_list_train[i]) if enthalpy_list_train[i] is not None else None,
        'elements': "-".join(element_list_train[i])
    })

validation_samples = []
for i in range(len(true_values)):
    avg_cn = cn_list_test[i]
    validation_samples.append({
        'index': i,
        'true_value': float(true_values[i]),
        'prediction': float(predictions[i]),
        'additional_info': float(add_info_list_test[i]) if add_info_list_test[i] is not None else None,
        'space_group': format_sg(sg_list_test[i]),
        'avg_coordination_number': float(avg_cn) if avg_cn is not None else None,
        'enthalpy_formation_atom': float(enthalpy_list_test[i]) if enthalpy_list_test[i] is not None else None,
        'elements': "-".join(element_list_test[i])
    })

# Add detailed sample data to JSON
serializable_info['training_samples'] = training_samples
serializable_info['validation_samples'] = validation_samples

# Also add summary statistics
sg_summary = {
    'training_set_size': len(sg_list_train),
    'validation_set_size': len(sg_list_test),
    'unique_space_groups_training': len(set(format_sg(sg) for sg in sg_list_train if sg is not None)),
    'unique_space_groups_validation': len(set(format_sg(sg) for sg in sg_list_test if sg is not None))
}
serializable_info['space_group_summary'] = sg_summary

with open(training_info_path, 'w') as f:
    json.dump(serializable_info, f, indent=4)