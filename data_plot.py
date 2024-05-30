# General
import numpy as np
import pandas as pd
import seaborn as snc
# Data visualization
import seaborn as sns
import matplotlib.pyplot as plt



# Plotting numerical data
def plot_numerical_data(dataset_df, columns = ['Age', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'RoomService'], saving_path=None):
    """
    Takes as inputs the dataset and the list of columns name corresponding to numerical values and plot histograms

    dataset_df --- (pandas.Dataframe)
    columns --- (list of str) 

    """
    fig, axes = plt.subplots(3, 2, figsize=(15, 15))
    axes = axes.flatten()

    for i, column in enumerate(columns):
        sns.histplot(dataset_df[column], bins=25, color='b', ax=axes[i])
        axes[i].set_title(column)
        axes[i].set_xlabel('')
        axes[i].set_ylabel('Count')

    if saving_path is not None:
        plt.savefig(saving_path)

    plt.tight_layout()
    plt.show()

## Plotting the correlation matrix

def correlation_matrix(dataset_df, saving_path=None):
    """
    Takes as inputs the dataset and computes the correlation matrix
    """
    numerical_columns = dataset_df.select_dtypes(include="number").columns
    if "Transported" not in numerical_columns:
        numerical_columns = numerical_columns.append(pd.Index(['Transported']))
    num_dataset = dataset_df[numerical_columns]
    correlation_matrix = num_dataset.corr()
    plt.figure(figsize=(10,8))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt = ".2f")
    plt.title('Correlation Matrix')

    if saving_path is not None:
        plt.savefig(saving_path)
    plt.tight_layout()
    plt.show()



# Additional plots
def plot_categorical_data(dataset_df, columns = ['HomePlanet', 'CryoSleep', 'Age', 'VIP', 'Destination'], saving_path = None):
    """
    Takes as imputs the dataset and the list of columns name corresponding to categorical data and plots the countplot

    dataset_df --- (pandas.Dataframe)
    columns --- (list of str) 

    """

    fig, axes = plt.subplots(3, 2, figsize=(15, 15))
    axes = axes.flatten()

    for i, column in enumerate(columns):
        sns.countplot(x=dataset_df[column], hue=dataset_df['Transported'], ax=axes[i])
        axes[i].set_title(column)
        axes[i].set_xlabel('')
        axes[i].set_ylabel('Count')
        if column == 'Age':
            # Set x-axis ticks for Age column
            major_ticks = np.arange(0, 110, 10)
            minor_ticks = np.arange(5, 100, 10)
            axes[i].set_xticks(major_ticks)
            axes[i].set_xticklabels([str(tick) if tick % 10 == 0 else '' for tick in major_ticks])
            axes[i].set_xticks(minor_ticks, minor=True)

    if saving_path is not None:
        plt.savefig(saving_path)

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    dataset_df = pd.read_csv("./Data/train.csv")
    plot_numerical_data(dataset_df, saving_path="./Data/numerical_data.png")

    correlation_matrix(dataset_df, saving_path="./Data/correlation_matrix.png")

    plot_categorical_data(dataset_df, saving_path="./Data/categorical_data.png")
