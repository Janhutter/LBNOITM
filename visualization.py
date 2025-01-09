import matplotlib.pyplot as plt
import json
import os
import pandas as pd
import numpy as np

def visualisation(models, top_ks, datasets, tasks):

    for task in tasks:
        experiments = {}
        # experiments = {model: {top_k: [{dataset: [f'{model}_top{top_k}{task}_oracle{i}_{dataset}' for i in range(top_k)]} for dataset in datasets] for top_k in top_ks} for model in models}
        for model in models:
            experiments[model] = {}
            for top_k in top_ks:
                experiments[model][top_k] = {}
                for dataset in datasets:
                    experiments[model][top_k][dataset] = [f'{model}_top{top_k}{task}_oracle{i}_{dataset}' for i in range(top_k)]

        metrics = ['EM', 'F1', 'Precision', 'Recall', 'M', 'Rouge-1', 'Rouge-2', 'Rouge-L']
        metrics = ['M', 'Recall']
        if len(models) == 1:
            baseline_names = ['Closed book', 'Oracle']
        else:
            baseline_names = []

        metric_labels = {
            'M': 'Match score',
            # 'EM': 'Exact Match',
            # 'F1': 'F1',
            # 'Precision': 'Precision',
            'Recall': 'Recall',
            # 'Rouge-1': 'Rouge-1',
            # 'Rouge-2': 'Rouge-2',
            # 'Rouge-L': 'Rouge-L',
        }

        map_datasets = {
            'kilt_nq': 'KILT NQ',
            'kilt_hotpotqa': 'KILT HotpotQA',
        }

        task_labels = {
            'relevant_not_correct': 'Hard distractor',
            'relevant': 'Relevant context',
            'random': 'Random distractor'
        }

        reverse_models = {
            'llama2_7bchat': 0,
            'llama2_7bchat': 1,
            'solar107b': 2,
            'mixtral7bchat': 3,
        }

        better_model_names = {
            'tinyllamachat': 'TinyLlama',
            'llama2_7bchat': 'Llama2-7B',
            'solar107b': 'Solar-10.7B',
            'mixtral7bchat': 'Mixtral-8x7B'
        }

        
        # print(colors)
        # colors = tab10.colors[1:]  # Skip the first color (blue)
        # color_cycle = cycler('color', colors)
        # plt.rc('axes', prop_cycle=color_cycle)

        # set legend color to the same as the lines

        # for a single model, the legend should be the model name + the baseline names
        if len(models) == 1:
            legend = [f'{task_labels[task]}'] + baseline_names
        else:
            # replace _ with ' '.
            legend_models = [model.replace('_', '-') for model in models]
            legend = legend_models + baseline_names

        results_per_topk = {top_k: [] for top_k in top_ks}
        for model in models:
            for top_k in experiments[model].keys():
                model_performance = {}
                for dataset in datasets:
                    performance = {}
                    for i, path in enumerate(experiments[model][top_k][dataset]):
                        with open(f'experiments/{path}/eval_dev_metrics.json') as f:
                            data = json.load(f)
                            for metric in metrics:
                                if metric not in performance:
                                    performance[metric] = []
                                performance[metric].append(data[metric])
                    model_performance[dataset] = performance
                    results_per_topk[top_k].append(model_performance)
    
        # adding baselines
        if len(models) == 1:
            baseline_res = {[dataset]: [] for dataset in datasets}
            for index, model in enumerate(models):
                for dataset in datasets:
                    baseline_experiments =[f'experiments/{model[index]}_closedbook_{dataset}/eval_dev_metrics.json',
                                                f'experiments/{model[index]}_oracle_{dataset}/eval_dev_metrics.json', ]
                    baseline_res[dataset].append(baseline_experiments)

                
                # if len(baseline_names) != len(baseline_experiments):
                #     raise ValueError('Number of baseline names should be equal to the number of baseline experiments')


                baseline_performances = [{dataset: {metric: [] for metric in metrics} for dataset in datasets} for _ in baseline_names]

                for dataset in datasets:
                    for i, baseline in enumerate(baseline_res[dataset]):
                        with open(baseline) as f:
                            data = json.load(f)
                            for metric in metrics:
                                if metric not in data:
                                    baseline_performances[i][dataset][metric] = [0.0]
                                else:
                                    baseline_performances[i][dataset][metric] = [data[metric]]

        else:
            baseline_performances = {}

        print(baseline_performances)


        # find minimum and maximum value for every metric 
        minimums = {dataset: {metric: np.inf for metric in metrics} for dataset in datasets}
        maximums = {dataset: {metric: -np.inf for metric in metrics} for dataset in datasets}
        # print(results_per_topk)
        for top_k in results_per_topk.values():
            for result_topk in top_k:
                for metric in metrics:
                    for dataset in datasets:
                        min_value = min(result_topk[dataset][metric])
                        max_value = max(result_topk[dataset][metric])
                        if min_value < minimums[dataset][metric]:
                            minimums[dataset][metric] = min_value
                        if max_value > maximums[dataset][metric]:
                            maximums[dataset][metric] = max_value
        line_colors = plt.colormaps.get_cmap('tab10').colors
        line_colors = line_colors[1:]
        baseline_colors = plt.colormaps.get_cmap('Set2').colors
        # colors = tab10.colors
        use_baselines = False
        if len(models) == 1 or use_baselines:
            for baseline in baseline_performances:
                for metric in metrics:
                    for dataset in datasets:
                        min_value = min(baseline[dataset][metric])
                        max_value = max(baseline[dataset][metric])
                        if min_value < minimums[dataset][metric]:
                            minimums[dataset][metric] = min_value
                        if max_value > maximums[dataset][metric]:
                            maximums[dataset][metric] = max_value

        for dataset in datasets:
            for minimum, maximum, metric in zip(minimums[dataset].values(), maximums[dataset].values(), metrics):
                minimums[dataset][metric] = minimum - 0.04
                maximums[dataset][metric] = maximum + 0.04

        if len(top_ks) == 1:
            fig, axs = plt.subplots(len(metrics), len(datasets) * len(top_ks), figsize=(3.5*len(top_ks), 4*len(metrics)))
            top_k = top_ks[0]
            for i, metric in enumerate(metrics):
                for result_topk in results_per_topk[0]:
                    axs[i].plot(range(1, len(result_topk[metric]) + 1), result_topk[metric], label=metric, marker='o', linestyle='-', linewidth=1, color=line_colors)
                axs[i].set_ylabel(f'{metric_labels[metric]}')
                axs[i].set_xlabel('Position of oracle document')
                # make sure grid is on every integer x-axis tick
                axs[i].grid(True, which='major')
                axs[i].set_xticks(range(1, len(result_topk[metric]) + 1))
                # set a title for every j. So every column has a title
                axs[0].set_title(f'Context length of {top_k}')
                axs[i].set_ylim(minimums[metric], maximums[metric])
                if len(models) == 1:
                    for k, baseline in enumerate(baseline_performances):
                        axs[i].plot(range(1, len(result_topk[metric]) + 1), [baseline[metric] for _ in range(len(result_topk[metric]))], label=baseline_names[k], linestyle='dashed', linewidth=1.5)

        elif len(metrics) > 1:
            fig, axs = plt.subplots(len(metrics), len(datasets) * len(top_ks), figsize=(3.5 * len(datasets) * len(top_ks), 4 * len(metrics)))

            for i, metric in enumerate(metrics):
                for g, dataset in enumerate(datasets):
                    for j, top_k in enumerate(top_ks):
                        # Group by dataset first, and then place the different top_k values next to each other
                        col_index = g * len(top_ks) + j
                        for s, result_topk in enumerate(results_per_topk[top_k]):
                            axs[i, col_index].plot(range(1, len(result_topk[dataset][metric]) + 1), result_topk[dataset][metric], label=metric, marker='o', linestyle='-', linewidth=1, color=line_colors[int(s / len(metrics))])
                            # set color for this line to the color of the model in the tab10 colormap
                            # axs[i, col_index].lines[-1].set_color(colors[s])
                        if col_index == 0:
                            axs[i, 0].set_ylabel(f'{metric_labels[metric]}', fontsize=15)
                        axs[i, col_index].set_xlabel('Position of oracle document', fontsize=15)
                        axs[i, col_index].grid(True, which='major')
                        axs[i, col_index].set_xticks(range(1, len(result_topk[dataset][metric]) + 1))
                        axs[0, col_index].set_title(f'{map_datasets[dataset]}\n Top-{top_k}', fontsize=16)
                        axs[i, col_index].set_ylim(minimums[dataset][metric], maximums[dataset][metric])
                        if len(models) == 1:

                            for k, baseline in enumerate(baseline_performances):
                                axs[i, col_index].plot(
                                    range(1, len(result_topk[dataset][metric]) + 1),
                                    [baseline[dataset][metric] for _ in range(len(result_topk[dataset][metric]))],
                                    label=baseline_names[k],
                                    linestyle='dashed',
                                    linewidth=1.5,
                                    color=baseline_colors[k]
                                )
        else: 
            fig, axs = plt.subplots(1, len(datasets) * len(top_ks), figsize=(3.5 * len(datasets) * len(top_ks), 4))
            line_colors = plt.colormaps.get_cmap('tab10').colors
            for g, dataset in enumerate(datasets):
                for j, top_k in enumerate(top_ks):
                    metric = metrics[0]
                    row_index = g * len(top_ks) + j
                    for s, result_topk in enumerate(results_per_topk[top_k]):
                        axs[row_index].plot(range(1, len(result_topk[dataset][metric]) + 1), result_topk[dataset][metric], label=metric, marker='o', linestyle='-', linewidth=1, color=line_colors[0])
                        # set color for this line to the color of the model in the tab10 colormap
                        # axs[i, col_index].lines[-1].set_color(colors[s])
                    axs[row_index].set_ylabel(f'{metric_labels[metric]}', fontsize=15)
                    axs[row_index].set_xlabel('Position of oracle document', fontsize=15)
                    axs[row_index].grid(True, which='major')
                    axs[row_index].set_xticks(range(1, len(result_topk[dataset][metric]) + 1))
                    axs[row_index].set_title(f'{map_datasets[dataset]}\n Top-{top_k}', fontsize=16)
                    axs[row_index].set_ylim(minimums[dataset][metric], maximums[dataset][metric])
                    if len(models) == 1:
                        for k, baseline in enumerate(baseline_performances):
                            axs[row_index].plot(
                                range(1, len(result_topk[dataset][metric]) + 1),
                                [baseline[dataset][metric] for _ in range(len(result_topk[dataset][metric]))],
                                label=baseline_names[k],
                                linestyle='dashed',
                                linewidth=1.5,
                                color=baseline_colors[k]
                            )

                    
            # reduce margins to the left and right of the whole plot drastically
            # plt.subplots_adjust(left=0.09, right=0.91)
        
        
        # add legend to the plot

        handles = []
        for i, model in enumerate(models):
            handles.append(plt.Line2D([0], [0], color=line_colors[i], label=better_model_names[model], linewidth=1.5))
        for i, baseline in enumerate(baseline_names):
            handles.append(plt.Line2D([0], [0], color=baseline_colors[i], label=baseline, linestyle='dashed', linewidth=1.5))
        # add the legend to the plot
        fig.legend(legend, loc='lower center', bbox_to_anchor=(0.5, 0.01), shadow=True, ncol=(len(models) * len(baseline_names) + len(models)), fontsize=15, handles=handles)



        # add more space between subplots
        plt.subplots_adjust(hspace=0.3)
        # add margin to top and bottom of the whole plot
        
        if len(models) == 1 and len(metrics) == 1:
            plt.subplots_adjust(top=0.8, bottom=0.28)
            plt.subplots_adjust(wspace=0.3)
        else:
            plt.subplots_adjust(top=0.92, bottom=0.15)
            plt.subplots_adjust(wspace=0.2)

        # add margin between the horizontal subplots 

        plt.subplots_adjust(left=0.05, right=0.98)

        if not os.path.isdir('figures'):
            os.mkdir('figures')

        if len(top_ks) == 1:
            plt.savefig(f'figures/{"_".join([model for model in models])}_{task}_{datasets}_top{top_ks[0]}.svg')
        else:
            plt.savefig(f'figures/{"_".join([model for model in models])}_{task}_{datasets}.svg')


if __name__ == '__main__':
    models = ['llama2_7bchat', 'solar107b', 'mixtral7bchat']
    # models = ['tinyllamachat']
    top_ks = [5, 10]
    dataset = ['kilt_nq','kilt_hotpotqa' ]
    #random, relevant_not_correct or relevant
    tasks = ['random', 'relevant_not_correct', 'relevant']
    #  
    visualisation(models, top_ks, dataset, tasks)