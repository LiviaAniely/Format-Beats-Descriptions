import os
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot_cross_models(results, save_path=None, title="Model Comparisons", sort_order=['vanilla(w/o CoT)', 'vanilla(w/ CoT)', 'ensemble_random(w/o CoT)', 'ensemble_random(w/ CoT)']):
    """
    The results of different templates across different models are plotted in a bar chart.
    """
    accuracies_dict = defaultdict(list)
    models = []

    for model_name, template2accuracy in results.items():
        models.append(model_name)
        for template, accuracy in template2accuracy.items():
            accuracies_dict[template].append(accuracy)

    # Sorting the accuracies_dict keys
    sorted_accuracies = {k: accuracies_dict[k] for k in sort_order if k in accuracies_dict}

    x = np.arange(len(models))  # the label locations
    num_methods = len(sorted_accuracies)
    width = 0.8 / num_methods  # the width of the bars, adjusted to fit all bars

    # Creating the plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Iterate over the dictionary to create bars
    bars = []
    for i, (method, accuracies) in enumerate(sorted_accuracies.items()):
        bar = ax.bar(x + (i - num_methods / 2) * width, accuracies, width, label=method)
        bars.append(bar)

    # Adding text for labels, title and custom x-axis tick labels, etc.
    ax.set_xlabel('Models')
    ax.set_ylabel('Accuracy')

    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2)

    # Adding value labels on bars
    def add_labels(bars):
        for bar_container in bars:
            for rect in bar_container.get_children():
                height = rect.get_height()
                ax.annotate('%.2f' % height,
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom')

    add_labels(bars)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    else:
        plt.show()

def plot_cross_datasets(results, model, category, save_path, sort_order):
    """
    The results of different datasets for a single model are plotted in a bar chart.
    """
    accuracies_dict = defaultdict(list)
    datasets = []

    for dataset_name, template2accuracy in results.items():
        datasets.append(dataset_name)
        for template, accuracy in template2accuracy.items():
            accuracies_dict[template].append(accuracy)

    # Sorting the accuracies_dict keys
    sorted_accuracies = {k: accuracies_dict[k] for k in sort_order if k in accuracies_dict}

    x = np.arange(len(datasets))  # the label locations
    num_methods = len(sorted_accuracies)
    width = 0.8 / num_methods  # the width of the bars, adjusted to fit all bars

    # Creating the plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Iterate over the dictionary to create bars
    bars = []
    for i, (method, accuracies) in enumerate(sorted_accuracies.items()):
        bar = ax.bar(x + (i - num_methods / 2) * width, accuracies, width, label=method)
        bars.append(bar)

    # Adding text for labels, title and custom x-axis tick labels, etc.
    ax.set_xlabel('Datasets')
    ax.set_ylabel('Accuracy')

    ax.set_title(f"{model} on {category}")
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2)

    # Adding value labels on bars
    def add_labels(bars):
        for bar_container in bars:
            for rect in bar_container.get_children():
                height = rect.get_height()
                ax.annotate('%.2f' % height,
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom')

    add_labels(bars)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    else:
        plt.show()
