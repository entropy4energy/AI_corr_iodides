# GNN Correction Project

This is a standalone version of the GNN correction model for predicting enthalpy formation corrections.

## Project Structure

```
.
├── Data/
│   ├── CCE_data.json          # Subset of training data (100 samples)
│   └── ele_data/
│       └── Aflow_e_data_GH.json  # Element properties data
├── Training/
│   ├── GNN_correction_lib2_eleCV.py  # Main training script
│   └── __init__.py
├── Util/
│   ├── graph_loader.py       # Graph building and data loading utilities
│   ├── visual_func.py        # Visualization utilities
│   ├── model.py              # GNN Model definition
│   └── __init__.py
├── predict.py                 # Inference script for calibrated CCE energy
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## How to Run

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   Note: For `torch-geometric`, you may need to follow specific installation instructions based on your CUDA version: [PyG Installation Guide](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html).

2. **Run Training**:
   ```bash
   python Training/GNN_correction_lib2_eleCV.py
   ```

3. **Inference (Prediction)**:
   After training, you can use `predict.py` to get the calibrated CCE formation energy for new structures:
   ```bash
   python predict.py
   ```
   You can integrate the `CCEPredictor` class into your own code as shown in `predict.py`.

## Configuration

The training configuration (target property, model parameters, test elements, etc.) can be modified directly in the `training_info` dictionary at the top of `Training/GNN_correction_lib2_eleCV.py`.

