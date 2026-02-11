import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import seaborn as sns
import pickle
import matplotlib.pyplot as plt
CONFIG = {
    "figure_size": (6, 6),  # Ensure 1:1 aspect ratio
    "scatter_color": "navy",  # Scatter color
    "scatter_alpha": 0.6,  # Transparency
    "scatter_size": 20,  # Scatter point size
    "fit_line_color": "orange",  # Fit line color
    "unity_line_color": "black",  # y=x reference line color
    "background_color": "#E6ECF4",  # Light blue background
    'font_size': 12,  # Font size for labels
    'legend_size': 10,  # Font size for legend
}

def plot_results(true_values, predicted_values,xlabel, ylabel, title, file_path):
    """
    Plot true vs. predicted values with unified visualization.
    """
    fig, ax = plt.subplots(figsize=CONFIG["figure_size"])
    ax.set_facecolor(CONFIG["background_color"])  # Set background color

    # Compute y=x reference line range
    x_vals = np.linspace(min(true_values), max(true_values), 100)
    ax.plot(x_vals, x_vals, '--', color=CONFIG["unity_line_color"], linewidth=1.5, label=r"$y = x$")

    # Compute linear fit
    slope, intercept = np.polyfit(true_values, predicted_values, 1)
    ax.plot(x_vals, slope * x_vals + intercept, color=CONFIG["fit_line_color"], linewidth=1.5, label="Linear Fit")

    # Scatter plot
    ax.scatter(true_values, predicted_values, color=CONFIG["scatter_color"], alpha=CONFIG["scatter_alpha"],
               s=CONFIG["scatter_size"], edgecolor='none')

    # Compute and display metrics
    rmse = np.sqrt(mean_squared_error(predicted_values, true_values))
    mae = mean_absolute_error(predicted_values, true_values)
    r2 = r2_score(predicted_values, true_values)
    metrics_text = f"R² = {r2:.3f}\nMAE = {mae:.3f}"
    ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes, fontsize=CONFIG['font_size'], verticalalignment='top', 
            bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="white"))

    # Set labels
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # Set aspect ratio and grid
    ax.set_aspect("equal", adjustable="datalim")  # Keep 1:1 aspect ratio
    plt.gca().set_box_aspect(1)  # Ensure the entire figure is 1:1
    ax.grid(True, linestyle="--", alpha=0.5)

    # Add legend
    ax.legend(loc="upper right", fontsize=CONFIG['font_size'], frameon=True, facecolor="white", edgecolor="black")

    # Save and display
    plt.savefig(file_path, dpi=300, bbox_inches="tight", pad_inches=0)
    plt.close()