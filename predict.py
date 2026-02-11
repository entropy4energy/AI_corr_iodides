import os
import sys
import torch
import json
import numpy as np
from torch_geometric.data import Data, Batch

# Add project root to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from Util.graph_loader import ele_df, StructureGraphBuilder
from Util.model import GNNModel

class CCEPredictor:
    def __init__(self, model_path, training_config_path=None):
        # Default config if not provided
        self.config = {
            'prop_list': [
                "atomic_mass", "atomic_number", "c6_gb", "density_PT",
                "electron_affinity_PT", "electronegativity_Allen",
                "electronegativity_Ghosh", "enthalpy_vaporization",
                "polarizability", "radii_Ghosh08", "radii_Pyykko",
                "radii_Slatter", "radius_PT", "radius_covalent_PT"
            ],
            'normalize': 'quantile',
            'graph_cutoff': 5.0,
            'fractional': True,
            'num_layers': 2,
            'hidden_dim': 64,
            'dropout': 0.1,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu'
        }
        
        # Load element properties
        self.ele_props = ele_df(prop_list=self.config['prop_list'], normalize=self.config['normalize'])
        self.builder = StructureGraphBuilder(self.ele_props, cutoff=self.config['graph_cutoff'], fractional=self.config['fractional'])
        
        # Initialize model
        # Note: We need input_dim which depends on the number of properties
        # For this specific model, input_dim = num_properties
        input_dim = len(self.config['prop_list'])
        self.model = GNNModel(
            input_dim=input_dim,
            hidden_dim=self.config['hidden_dim'],
            output_dim=1,
            num_layers=self.config['num_layers'],
            dropout=self.config['dropout'],
            num_properties=input_dim
        ).to(self.config['device'])
        
        # Load weights
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.config['device']))
            self.model.eval()
            print(f"Model loaded from {model_path}")
        else:
            print(f"Warning: Model path {model_path} not found. Prediction will use random weights.")

    def predict(self, structure_data, enthalpy_formation_atom, avg_coordination_number):
        """
        Predict calibrated CCE formation energy.
        
        Args:
            structure_data: dict with 'Lattice Matrix', 'Atomic Positions', 'Elements'
            enthalpy_formation_atom: float, raw enthalpy formation
            avg_coordination_number: float, average coordination number
            
        Returns:
            calibrated_cce: float
            correction: float
        """
        # Build graph
        graph = self.builder.build_graph(structure_data)
        
        # Prepare input1 (element properties)
        elements = [x for x in structure_data['Elements'] if x != "I"]
        element_props = self.ele_props.loc[:, elements].T.values
        graph.input1 = torch.tensor(element_props, dtype=torch.float)
        
        # Prepare input2 (CN and Enthalpy)
        graph.input2 = torch.tensor([avg_coordination_number, enthalpy_formation_atom], dtype=torch.float)
        
        # Create Batch for single graph
        batch = Batch.from_data_list([graph]).to(self.config['device'])
        
        with torch.no_grad():
            correction = self.model(batch).item()
            
        calibrated_cce = enthalpy_formation_atom + correction
        return calibrated_cce, correction

if __name__ == "__main__":
    # Example usage
    model_file = "Output/Training/best_model.pth" # Update after training
    predictor = CCEPredictor(model_file)
    
    # Example data (dummy)
    example_structure = {
        "Lattice Matrix": [[4.0, 0, 0], [0, 4.0, 0], [0, 0, 4.0]],
        "Atomic Positions": [[0, 0, 0], [0.5, 0.5, 0.5]],
        "Elements": ["Li", "Li"]
    }
    enthalpy = -1.5
    cn = 6.0
    
    cce, corr = predictor.predict(example_structure, enthalpy, cn)
    print(f"\nExample Prediction:")
    print(f"Raw Enthalpy: {enthalpy:.4f} eV/atom")
    print(f"Predicted Correction: {corr:.4f} eV/atom")
    print(f"Calibrated CCE Energy: {cce:.4f} eV/atom")

