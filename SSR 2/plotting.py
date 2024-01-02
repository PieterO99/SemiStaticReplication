import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def visualize_fit(S, predictions, option_values, stock_values, time, monitored_stock, monitored_prices):
    plt.figure()
    suffix = 'th' if 10 <= time % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(time % 10, 'th')
    plt.scatter(stock_values * S, option_values * S, label=f'Option value at {time}-{suffix} monitoring date via RLNN',
                color='b', s=0.4)

    # plt.scatter(stock_values * S, predictions * S,
    #             label=f'Regressed option value at maturity', color='r', s=0.4)

    mask = np.where(monitored_stock[time] > 0)[0]
    plt.scatter(monitored_stock[time][mask], monitored_prices[time][mask],
                label=f'Option value at {time}-{suffix} monitoring date via Binomial Model', color='m', s=0.5)

    plt.xlabel('Stock Value')
    plt.ylabel('Option Value')
    plt.legend()
    plt.savefig(f'Fit_NN{time}.png', dpi=200, transparent=True)
    plt.show()


def plot_option_vs_hidden_nodes(csv_file, target_value):
    df = pd.read_csv(csv_file)
    grouped_df = df.groupby('hidden_nodes')['initial_option_value'].mean().reset_index()

    # Plot 'initial_option_value' as a function of 'hidden_nodes'
    plt.figure(figsize=(10, 6))
    plt.plot(grouped_df['hidden_nodes'], grouped_df['initial_option_value'], color='blue', marker='o')

    y_ticks = np.arange(2, grouped_df['initial_option_value'].max() + 0.25, 0.25)
    plt.yticks(y_ticks)

    x_ticks = np.arange(0, grouped_df['hidden_nodes'].max() + 8, 8)
    plt.xticks(x_ticks)

    plt.axhline(y=target_value, color='red', linestyle='--', label=f'Target Value: {target_value}')

    plt.title('Initial Option Value vs Hidden Nodes')
    plt.xlabel('Hidden Nodes')
    plt.ylabel('Initial Option Value')
    plt.grid(True)
    plt.legend()
    plt.show()


def plot_option_vs_learning_rate(csv_file, target_value):
    csv_data = pd.read_csv(csv_file)

    data_hidden_32 = csv_data[csv_data['hidden_nodes'] == 32]
    l3_values_32 = data_hidden_32['l3']
    initial_option_values_32 = data_hidden_32['initial_option_value']
    N_values_32 = data_hidden_32['N']

    # Create a scatter plot for hidden_nodes=32
    plt.figure(figsize=(10, 6))
    colors_32 = plt.cm.viridis(N_values_32 / max(N_values_32))
    plt.scatter(l3_values_32, initial_option_values_32, s=0.9, c=colors_32, alpha=0.8,
                label='Option value estimate (hidden_nodes=32)')
    plt.axhline(y=target_value, color='red', linestyle='--', label=f'Target Value: {target_value}')
    y_ticks = np.arange(2, initial_option_values_32.max() + 0.25, 0.25)
    plt.yticks(y_ticks)
    plt.xlabel('Learning Rate')
    plt.ylabel('Initial Option Value')
    plt.title('Initial Option Value vs Learning Rate (hidden_nodes=32)')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Extract relevant data from the DataFrame for hidden_nodes=64
    data_hidden_64 = csv_data[csv_data['hidden_nodes'] == 64]
    l3_values_64 = data_hidden_64['l3']
    initial_option_values_64 = data_hidden_64['initial_option_value']
    N_values_64 = data_hidden_64['N']

    # Create a scatter plot for hidden_nodes=64
    plt.figure(figsize=(10, 6))
    colors_64 = plt.cm.viridis(N_values_64 / max(N_values_64))
    plt.scatter(l3_values_64, initial_option_values_64, s=0.9, c=colors_64, alpha=0.8,
                label='Option value estimate (hidden_nodes=64)')
    plt.axhline(y=target_value, color='red', linestyle='--', label=f'Target Value: {target_value}')
    y_ticks = np.arange(2, initial_option_values_64.max() + 0.25, 0.25)
    plt.yticks(y_ticks)
    plt.xlabel('Learning Rate')
    plt.ylabel('Initial Option Value')
    plt.title('Initial Option Value vs Learning Rate (hidden_nodes=64)')
    plt.legend()
    plt.grid(True)
    plt.show()
