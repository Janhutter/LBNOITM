import matplotlib.pyplot as plt
import numpy as np
import json
import pandas as pd
import ast
import matplotlib
import matplotlib.colors as mcolors

def visualize_attention(top_k, tasks, dataset, model_name):
    data = {'task': [], 'model': [], 'dataset': [], 'attention_first': [], 'attention_last': [], 'attention_middle': [], 'pos': []}
    for task in tasks:
        for pos in range(top_k):
            with open(f'experiments/attention_{model_name}_top{top_k}{task}_oracle{pos}_{dataset}/eval_dev_out.json') as f:
                attention = json.load(f)
                for row in attention:
                    data['task'].append(task)
                    data['model'].append(model_name)
                    data['dataset'].append(dataset)
                    real_data = row['response']
                    real_data = ast.literal_eval(real_data)
                    data['attention_first'].append(real_data[1][0])
                    data['attention_last'].append(real_data[1][2])
                    data['attention_middle'].append(real_data[1][1])
                    data['pos'].append(pos)  # Adjust position to start from 1

    data = pd.DataFrame(data)
    # Check what is the highest attention value inside attention_first, attention_last, attention_middle and give the index
    # but only of the second to second to last values inside every list inside the attention values
    data['highest_pos_first'] = data['attention_first'].apply(lambda x: np.argmax(x[1:-1]))
    data['highest_pos_last'] = data['attention_last'].apply(lambda x: np.argmax(x[1:-1]))
    data['highest_pos_middle'] = data['attention_middle'].apply(lambda x: np.argmax(x[1:-1]))

    # print(data.head())

    for task in tasks:
        fig, ax = plt.subplots(1, 3, figsize=(20, 10))

        # Define a consistent color map
        colors = plt.get_cmap('tab10').colors

        for i, attention in enumerate(['first', 'middle', 'last']):
            grouped = data[(data['task'] == task)].groupby('pos')[f'highest_pos_{attention}'].value_counts().unstack().fillna(0)
            grouped_pct = grouped.div(grouped.sum(axis=1), axis=0) * 100  # Convert to percentage
            grouped_pct.index = grouped_pct.index + 1

            # Assign colors dynamically based on the number of columns in grouped_pct
            column_colors = {col: colors[col] for i, col in enumerate(grouped_pct.columns)}

            grouped_pct.plot(kind='bar', stacked=True, ax=ax[i], legend=False, color=[colors[col] for col in grouped_pct.columns])

            ax[i].set_title(f'{attention} attention layer', fontsize=22)
            ax[i].set_xlabel('Position of oracle document in context', fontsize=20)
            ax[i].set_ylabel('Percentage of attention attributed to document', fontsize=20)
            ax[i].set_ylim([0, 100])  # Ensure y-axis is between 0 and 100 for percentage
            ax[i].set_xticklabels([i for i in range(1, top_k + 1)], rotation=0, fontsize=20)
            ax[i].yaxis.set_tick_params(labelsize=18)

        # Add legend under subplots
        handles = [matplotlib.patches.Patch(color=color, label=(col + 1)) for col, color in column_colors.items()]
        fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 0.1), ncol=top_k, title='Document Position', fontsize=18, title_fontsize=18)

        plt.tight_layout(rect=[0, 0, 1, 0.98])
        plt.subplots_adjust(bottom=0.2)  # Add more space to the bottom
        plt.savefig(f'figures/attention_{model_name}_top{top_k}_{task}_{dataset}.svg')
        plt.show()

if __name__ == '__main__':
    top_k = 5
    tasks = ['random', 'relevant_not_correct', 'relevant']
    dataset = 'kilt_hotpotqa'
    model_name = 'tinyllamachat'
    visualize_attention(top_k, tasks, dataset, model_name)
