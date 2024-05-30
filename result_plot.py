# Accuracy Plot
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def read_file(filename, columns):
    data = []
    with open(filename, "r") as f:
        for line in f:
            parts = line.strip().split("     ")
            if len(parts) >= len(columns):
                row = [parts[0]] + [round(float(parts[i]),3) for i in range(1, len(columns))]
                data.append(row)
    return pd.DataFrame(data, columns=columns)




def plot(df,title, x_label, y_label, annotate = True, saving_path = None):
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Define the bar width and positions
    num_columns = len(df.columns)
    bar_width = 0.8 / num_columns  # Divide the total width into equal parts for each bar
    index = np.arange(len(df))

    bars = []
    for i, column in enumerate(df.columns[1:], start=1):
        bars.append(ax.bar(index +(i - num_columns/2) * bar_width, np.array(df[column]), bar_width, label = column))



    # Add text for the labels, title and custom x-axis tick labels, etc.
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.set_xticks(index)
    ax.set_xticklabels(df[df.columns[0]])
    ax.legend()

    # Add the accuracy values on top of the bars
    if annotate:
        for bars_set in bars:
            for bar in bars_set:
                height = bar.get_height()
                ax.annotate(f'{height:.3f}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom')

    if saving_path is not None:
        plt.savefig(saving_path)

    # Show the plot
    plt.tight_layout()
    plt.show()



if __name__ == '__main__':
    accuracy_data = read_file("./result/accuracy_data.txt",  ["Classifier", "Training Accuracy", "Validation Accuracy"])
    plot(accuracy_data, title = "Training and Validation Accuracies for Classical Classifier", x_label = "Classifier", y_label = "Accuracy", saving_path = "./result/accuracy_classical.png")


    time_data = read_file("./result/time_data.txt", ["Classifier", "Time"])
    plot(time_data, title = "Running Time for Classifier", x_label = "Classifier", y_label = "Time (s)", saving_path = "./result/time_classical.png")
