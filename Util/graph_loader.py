# -*- coding: utf-8 -*-
"""
Created by: Guangshuai (Jerry) Han @S4E Lab
Year: 2025
Affiliation: Johns Hopkins University
Purpose: This code is designed to process and normalize element data for the purpose of
         chemical composition feature extraction.
         It includes functionalities for:
         - Loading and normalizing element data.
         - Encoding chemical elements based on their properties.
         - Generating statistical features for machine learning tasks.
"""

import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split

import os

# Get the path to the current file's directory
current_dir = os.path.dirname(os.path.abspath(__file__))
# Define the path to the element data file relative to this file
json_path = os.path.join(current_dir, '../Data/ele_data/Aflow_e_data_GH.json')

with open(json_path, 'r') as file:
    ele_data = json.load(file)

e_list = list(ele_data.keys())
#remove the "AAA_notes"
e_list.remove("AAA_notes")
def get_atomnum(name):
    """
    Retrieve the atomic number for a given element name using the original method.
    """
    try:
        return ele_data[name]['atomic_number']
    except:
        return None

# function that create a dataframe from element data
# extract the specific properties from the json element data
# fill the null values with the mean of the column and nomalize the data

def ele_df(prop_list=['atomic_number', 'atomic_mass', 'density', 
                'electronegativity_Allen', 'radii_Ghosh08', 'radii_Pyykko',
                'radii_Slatter', 'radius_covalent','polarizability',
                'thermal_conductivity_25C'], total_e_list=e_list, normalize='Z-score'):

    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import quantile_transform

    # Initialize element dataframe and missing value tracker
    ele_df = pd.DataFrame(columns=prop_list, index=total_e_list)
    missing_report = pd.DataFrame(0, index=total_e_list, columns=prop_list)

    # Load data from element dictionary
    for ele in total_e_list:
        for prop in prop_list:
            try:
                ele_df.loc[ele, prop] = ele_data[ele][prop]
            except:
                ele_df.loc[ele, prop] = None
                missing_report.loc[ele, prop] = 1

    # Convert to numeric and detect missing values
    ele_df = ele_df.apply(pd.to_numeric, errors='coerce')
    missing_before = ele_df.isna()

    # Fill missing values with column mean and summarize fill counts
    ele_df = ele_df.apply(lambda x: x.fillna(x.mean()), axis=0)
    filled_report = missing_before & ~ele_df.isna()
    fill_counts = filled_report.sum(axis=1)
    fill_summary = fill_counts[fill_counts > 0].sort_values(ascending=False)

    # Print missing fill report
    print("\n[Missing Value Fill Report]")
    print("Number of elements with filled values:", len(fill_summary))
    #print(fill_summary.to_string())

    # Apply normalization based on method selected
    if normalize == 'Z-score':
        ele_df = (ele_df - ele_df.mean()) / ele_df.std()

    elif normalize == 'min-max':
        ele_df = (ele_df - ele_df.min()) / (ele_df.max() - ele_df.min())

    elif normalize == 'min-max-0.1-0.9':
        ele_df = 0.1 + 0.8 * (ele_df - ele_df.min()) / (ele_df.max() - ele_df.min())

    elif normalize == 'log-zscore':
        ele_df = np.log(ele_df + 1e-6)
        ele_df = (ele_df - ele_df.mean()) / ele_df.std()

    elif normalize == 'quantile':
        ele_df = pd.DataFrame(quantile_transform(ele_df, n_quantiles=100, output_distribution='uniform', copy=True),
                              index=ele_df.index, columns=ele_df.columns)

    elif normalize == 'none':
        pass  # No normalization applied

    else:
        raise ValueError("Unsupported normalization method. Choose from 'Z-score', 'min-max', 'min-max-0.1-0.9', 'log-zscore', 'quantile', or 'none'.")

    # Transpose the dataframe: elements as columns, properties as rows
    ele_df = ele_df.transpose()

    return ele_df



# define a GNN encoder to convert structure data to a graph
#define a function to extract the lattice matrix, atomic positions and elements from the structure data and convert them to a graph
import torch
import torch_geometric
from torch_geometric.data import Data

class StructureGraphBuilder:
    def __init__(self, ele_df, cutoff=5.0, fractional=True, use_global_attr=True):
        """
        Initialize the graph builder for structure data.

        Args:
            ele_df (dict): Dictionary mapping elements to embedding vectors.
            cutoff (float): Distance threshold for connecting edges.
            fractional (bool): If True, atomic positions are in fractional coordinates.
            use_global_attr (bool): If True, add a global graph attribute `u`.
        """
        self.ele_df = ele_df
        self.cutoff = cutoff
        self.fractional = fractional
        self.use_global_attr = use_global_attr

    def build_graph(self, structure_data, global_attr=1.0):
        """
        Build a graph from structure data.

        Args:
            structure_data (dict): Must include "Lattice Matrix", "Atomic Positions", "Elements".
            global_attr (float): Optional global graph-level attribute.

        Returns:
            torch_geometric.data.Data: Graph data object.
        """
        # Extract structure data
        lattice_matrix = np.array(structure_data["Lattice Matrix"])
        atomic_positions = np.array(structure_data["Atomic Positions"])
        elements = np.array(structure_data["Elements"])

        # Convert fractional to cartesian coordinates if needed
        if self.fractional:
            atomic_positions = atomic_positions @ lattice_matrix

        num_atoms = len(elements)

        node_features = []
        edge_index = []
        edge_distance = []
        edge_angle = []
        edge_weight = []

        # Create node feature matrix
        for ele in elements:
            node_features.append(self.ele_df[ele])

        # Create edges based on cutoff and compute attributes
        for i in range(num_atoms):
            for j in range(i + 1, num_atoms):
                vec_i = atomic_positions[i]
                vec_j = atomic_positions[j]
                diff_vec = vec_j - vec_i
                dist = np.linalg.norm(diff_vec)

                if 1e-6 < dist < self.cutoff:
                    # Bi-directional edge
                    for a, b in [(i, j), (j, i)]:
                        edge_index.append([a, b])
                        edge_distance.append(dist)
                        edge_weight.append(1.0 / dist)

                        # Angle = angle between i and j from origin
                        angle = np.arccos(np.dot(vec_i, vec_j) /
                                          (np.linalg.norm(vec_i) * np.linalg.norm(vec_j)))
                        edge_angle.append(angle)

        # Construct PyTorch Geometric Data object
        data = Data(
            x=torch.tensor(node_features, dtype=torch.float32),
            edge_index=torch.tensor(edge_index, dtype=torch.long).T.contiguous(),
            edge_attr=torch.tensor(edge_distance, dtype=torch.float32),
            edge_weight=torch.tensor(edge_weight, dtype=torch.float32),
            edge_angle=torch.tensor(edge_angle, dtype=torch.float32),
            pos=torch.tensor(atomic_positions, dtype=torch.float32)
        )
        #print everythin

        # Optionally add global attribute
        if self.use_global_attr:
            data.u = torch.tensor([global_attr], dtype=torch.float32)

        return data