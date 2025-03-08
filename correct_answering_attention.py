import matplotlib.pyplot as plt
import numpy as np
import json
import pandas as pd
import ast
import re
from sklearn.metrics import confusion_matrix

def normalize(string):
    # remove parentheses, brackets and punctuation including . , ! : ; 
    string = re.sub(r'[()\[\]{}\‘\’.,!:;"/\\\'|<>]', '', string)
    string = string.lower().strip()
    string = re.sub(r'\b(a|an|the)\b', '', string)
    return string

def match_criteria(labels, entry):
    """
    Matches a list labels to a string entry
    """
    entry = normalize(entry)
    for label in labels:
        label = normalize(label)
        pattern = r"\b" + re.escape(label) + r"\b"
        if re.search(pattern, entry):
            return True
    return False

def correct_answering_attention(top_k, tasks, datasets, model_name):
    data = {'task': [], 'model': [], 'dataset': [], 'attention_first': [], 'attention_last': [], 'attention_middle': [], 'pos': [], 'correct_answer': []}
    for task in tasks:
        for dataset in datasets:
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
                        data['pos'].append(pos)
                        answer = real_data[0]
                        label = row['label']
                        data['correct_answer'].append(match_criteria(label, answer))

    small_text = 21
    big_text = 22
    title = 24
    data = pd.DataFrame(data)
    # Check what is the highest attention value inside attention_first, attention_last, attention_middle and give the index
    # but only of the second to second to last values inside every list inside the attention values
    # data['highest_pos_first'] = data['attention_first'].apply(lambda x: np.argmax(x[1:-1]))
    data['highest_pos_last'] = data['attention_last'].apply(lambda x: np.argmax(x[1:-1]))
    data['highest_pos_middle'] = data['attention_middle'].apply(lambda x: np.argmax(x[1:-1]))
    # print('correct:', len(data[data['correct_answer']]))
    # print('incorrect:', len(data[~data['correct_answer']]))
    sliced = data[data['task'] == 'relevant_not_correct'].groupby(['pos'])
    for group in sliced:
        print(group[0][0])
        print('correct:', len(group[1][group[1]['correct_answer']]))
        print('incorrect:', len(group[1][~group[1]['correct_answer']]))

    # Create a figure and axes array
    for task in tasks:
        fig, ax = plt.subplots(2, 4, figsize=(21, 10))
        
        vmin, vmax = 0, 1  # Setting the color limits for the colormap
        for i, dataset in enumerate(datasets):
            for j, attention in enumerate(['middle', 'last']):
                index = int(f'{i}{j}', 2)
                values = {}
                for pos in range(top_k):
                    grouped = data[(data['task'] == task) & (data['correct_answer']) & (data['pos'] == pos) & (data['dataset'] == dataset)][f'highest_pos_{attention}'].sample(n=50)
                    value = grouped.value_counts().to_dict()
                    # add 1 to every key of the dict
                    value = {k + 1: v for k, v in value.items()}
                    values[pos + 1] = value
                grouped = pd.DataFrame(values)
                grouped.fillna(0, inplace=True)
                # make sure the y axis is sorted
                grouped = grouped.sort_index()
                # fill the other positions with 0
                # add rows for the other positions
                for pos in range(1, top_k + 1):
                    if pos not in grouped.index:
                        grouped.loc[pos] = 0

                grouped = grouped.sort_index()
                # Calculate the confusion matrix
                grouped = grouped.div(grouped.sum(axis=0), axis=1)
                grouped = grouped.fillna(0)
                cax = ax[0, index].imshow(grouped, cmap='viridis', interpolation='nearest', vmin=vmin, vmax=vmax)
                ax[0, index].set_title(f'{attention} attention layer', fontsize=big_text, fontweight='bold')
                ax[0, index].set_xlabel('Position of oracle document', fontsize=small_text)
                ax[0, index].set_ylabel('Highest attention value', fontsize=small_text)
                ax[0, index].set_xticks(np.arange(top_k))
                ax[0, index].set_yticks(np.arange(top_k))
                ax[0, index].set_xticklabels(np.arange(1, top_k + 1), fontsize=small_text)
                ax[0, index].set_yticklabels(np.arange(1, top_k + 1), fontsize=small_text)

        # Plotting the confusion matrix for incorrect answers
        for i, dataset in enumerate(datasets):
            for j, attention in enumerate(['middle', 'last']):
                index = int(f'{i}{j}', 2)
                values = {}
                for pos in range(top_k):
                    grouped = data[(data['task'] == task) & (~data['correct_answer']) & (data['pos'] == pos) & (data['dataset'] == dataset)][f'highest_pos_{attention}'].sample(n=50)
                    value = grouped.value_counts().to_dict()
                    # add 1 to every key of the dict
                    value = {k + 1: v for k, v in value.items()}
                    values[pos + 1] = value
                grouped = pd.DataFrame(values)
                grouped.fillna(0, inplace=True)
                # make sure the y axis is sorted
                grouped = grouped.sort_index()
                # fill the other positions with 0
                # add rows for the other positions
                for pos in range(1, top_k + 1):
                    if pos not in grouped.index:
                        grouped.loc[pos] = 0

                grouped = grouped.sort_index()
                # Calculate the confusion matrix
                grouped = grouped.div(grouped.sum(axis=0), axis=1)
                grouped = grouped.fillna(0)
                ax[1, index].imshow(grouped, cmap='viridis', interpolation='nearest', vmin=vmin, vmax=vmax)
                ax[1, index].set_title(f'{attention} attention layer', fontsize=big_text, fontweight='bold')
                ax[1, index].set_xlabel('Position of oracle document', fontsize=small_text)
                ax[1, index].set_ylabel('Highest attention value', fontsize=small_text)
                ax[1, index].set_xticks(np.arange(top_k))
                ax[1, index].set_yticks(np.arange(top_k))
                ax[1, index].set_xticklabels(np.arange(1, top_k + 1), fontsize=small_text)
                ax[1, index].set_yticklabels(np.arange(1, top_k + 1), fontsize=small_text)

        # Add row titles for the subplots
        ax[0, 0].set_ylabel('Answering correctly\n \n Highest attention value', fontsize=small_text)
        ax[1, 0].set_ylabel('Answering incorrectly \n \n Highest attention value', fontsize=small_text)

        # Adjust the layout to make room for the color bar but less than [0, 0, 0.9, 0.96]
        # plt.tight_layout(rect=[0.99, 0.8, 0.96, 0.961])
        # reduce figure margin top
        plt.subplots_adjust(left=0.055, right=0.88, top=0.9, bottom=0.1)

        # remove space between the subplots horizontally
        plt.subplots_adjust(wspace=0.2, hspace=0.4)
        # Adding a single color bar for the entire figure
        cbar_ax = fig.add_axes([0.92, 0.15, 0.01, 0.7])  # [left, bottom, width, height]
        cbar = fig.colorbar(cax, cax=cbar_ax)
        cbar.set_label('Percentage', fontsize=small_text)

        # ax[0, i].yaxis.set_tick_params(labelsize=16)

        plt.suptitle(f'KILT NQ                                                                KILT HotpotQA', fontsize=title, fontweight='bold')

        # Format the color bar ticks to show percentages
        cbar.set_ticks([0, 0.2, 0.4, 0.6, 0.8, 1])
        cbar.set_ticklabels(['0%', '20%', '40%', '60%', '80%', '100%'], fontsize=small_text)

        # less padding left and right of the figure
        # plt.subplots_adjust(left=0.05, right=1)

        plt.show()
        plt.savefig(f'figures/confusion_matrix_{model_name}_{task}_{dataset}.svg')

if __name__ == '__main__':
    top_k = 5
    tasks = ['random', 'relevant_not_correct', 'relevant']
    # dataset = 'kilt_hotpotqa'
    datasets = ['kilt_nq', 'kilt_hotpotqa']
    model_name = 'llama2_7bchat'
    correct_answering_attention(top_k, tasks, datasets, model_name)
