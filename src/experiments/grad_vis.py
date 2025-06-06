import matplotlib.pyplot as plt
import torch
import numpy as np

def plot_gradients_histogram(model, step, save_path=None):
    plt.figure(figsize=(12, 8))
    for name, param in model.named_parameters():
        if param.grad is not None:
            plt.hist(param.grad.data.cpu().numpy().flatten(), bins=50, alpha=0.7, label=name)
    plt.title(f'Gradient Histogram at Step {step}')
    plt.xlabel('Gradient Value')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()
