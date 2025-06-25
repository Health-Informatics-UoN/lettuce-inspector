"""
Simple example plots which can be logged to the mlflow tracking uri.
"""
from pathlib import Path 
import pandas as pd 
from matplotlib import pyplot as plt 
import seaborn as sns 


def plot_boxplot(
    data: pd.Series | pd.DataFrame, 
    palette: str = None, 
    x_label: str = None, 
    y_label: str = None, 
    plot_scatter: bool = True, 
    path_to_save_file: Path = None 
): 
    """Plot boxplot of numerical data"""
    fig, ax = plt.subplots()
    
    showfliers = True 
    if plot_scatter: 
        showfliers = False
    
    sns.boxplot(
        data, 
        orient="v",
        palette=palette,  
        ax=ax, 
        showfliers=showfliers 
    )
    
    if plot_scatter: 
        sns.stripplot(data, color='red', alpha=0.6, size=4)
    
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    if path_to_save_file: 
        fig.savefig(path_to_save_file, dpi=300)
    
    return fig, ax 


def plot_violinplot(
    data: pd.Series | pd.DataFrame, 
    palette: str = None, 
    x_label: str = None, 
    y_label: str = None, 
    path_to_save_file: Path = None 
): 
    """Plot violinplot of numerical data"""
    fig, ax = plt.subplots()
    
    sns.violinplot(
        data, 
        orient="v",
        palette=palette,  
        inner="points", 
        ax=ax
    )
    
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    if path_to_save_file: 
        fig.savefig(path_to_save_file, dpi=300)
    
    return fig, ax 