import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def table(models, top_ks, dataset, tasks):

    results_table = []

    baseline_names = ['closedbook', 'oracle']

    for model in models:
        for top_k in top_ks:
            for task in tasks:
                for position in range(top_k):
                    path = f'{model}_top{top_k}{task}_oracle{position}_{dataset}'
                    try:
                        with open(f'experiments/{path}/eval_dev_metrics.json') as f:
                            data = json.load(f)
                            results_table.append({
                                'Model': model,
                                'Top-K': top_k,
                                'Context Type': task,
                                'Position': position,
                                'M': data['M'],
                                'Recall': data['Recall']
                            })
                    except:
                        results_table.append({
                            'Model': model,
                            'Top-K': top_k,
                            'Context Type': task,
                            'Position': position,
                            'M': 0,
                            'Recall': 0
                        })


    for model in models:
        for baseline in baseline_names:
            path = f'experiments/{model}_{baseline}_{dataset}/eval_dev_metrics.json'
            try:
                with open(path) as f:
                    data = json.load(f)
                    results_table.append({
                        'Model': model,
                        'Top-K': 0,
                        'Context Type': baseline,
                        'Position': 0,
                        'M': data['M'],
                        'Recall': data['Recall']
                    })
            except:
                results_table.append({
                    'Model': model,
                    'Top-K': 0,
                    'Context Type': baseline,
                    'Position': 0,
                    'M': 0,
                    'Recall': 0
                })

    # Create and save the table
    df = pd.DataFrame(results_table)


    # make sure they are sorted by model in alphabetical order, with lower topk first and also context type first
    df = df.sort_values(by=['Model', 'Top-K', 'Context Type'])
    # round the values to 2 decimals

    new_models = {
        'tinyllamachat': 'TinyLlamaChat',
        'llama2_7bchat': 'Llama2-7b-Chat',
        'solar107b': 'Solar-107b',
        'mixtral7bchat': 'MixTRAL-7b-Chat'
    }

    new_tasks = {
        'random': 'Random Distractor',
        'relevant_not_correct': 'Hard Distractor',
        'relevant': 'Relevant Distractor',
        'oracle': 'Oracle',
        'closedbook': 'Closed Book'
    }   

    df['Model'] = df['Model'].map(new_models)
    models = [new_models[model] for model in models]

    df['Context Type'] = df['Context Type'].map(new_tasks)
    tasks = [new_tasks[task] for task in tasks]

    print(df['Context Type'].unique())
    # create 1 row for every model and combine the results like: topk_task_metric_position. So we get a lot of columns
    column_names = ['Model']
    for top_k in top_ks:
        for task in tasks:
            for metric in ['M', 'Recall']:
                for position in range(top_k):
                    column_names.append(f'{top_k}\_{task}\_{metric}\_{position}')

    # add oracle and closed book
    for baseline in baseline_names:
        for metric in ['M', 'Recall']:
            column_names.append(f'0\_{baseline}\_{metric}\_0')


    
    # create a new dataframe with the columns
    df2 = pd.DataFrame(columns=column_names)
    for model in models:
        for top_k in top_ks:
            for task in tasks:
                for position in range(top_k):
                    for metric in ['M', 'Recall']:
                        value = df[(df['Model'] == model) & (df['Top-K'] == top_k) & (df['Context Type'] == task) & (df['Position'] == position)][metric].values[0]
                        df2.loc[model, f'{top_k}\_{task}\_{metric}\_{position}'] = value
    # make sure that model name is filled in
    df2['Model'] = df2.index

    # add oracle and closed book
    for model in models:
        for baseline in new_tasks['closedbook'], new_tasks['oracle']:
            for metric in ['M', 'Recall']:
                value = df[(df['Model'] == model) & (df['Context Type'] == baseline)][metric].values[0]
                df2.loc[model, f'0\_{baseline}\_{metric}\_0'] = value

    # make sure that every model has a row per topk. So every model has 2 rows. You can do this by taking the 


    # remove the columns that are not needed anymore
    # df = df.drop(columns=['M', 'Recall', 'Position', 'Context Type'])
    # drop duplicates 
    
    # print(df.head(20))

    # round the values to 3 decimals
    df2 = df2.round(3)
    # remove trailing 0s but let them remain floats
    df2 = df2.map(lambda x: "%.3f" % x if isinstance(x, float) else x)

    #drop columns with nan
    # drop old baseline columns
    df2 = df2.drop(columns=['0\_closedbook\_M\_0', '0\_closedbook\_Recall\_0', '0\_oracle\_M\_0', '0\_oracle\_Recall\_0'])
    # df2 = df2.dropna(axis=1, how='all')

    # add a column named '' before the first column 
    # df2.insert(0, '', '')

    df2.to_latex(f'figures/table_{dataset}.tex', index=False)

    with open(f'figures/table_{dataset}.tex', 'r') as file:
        data = file.readlines()
        # print(data)

    first_line = ['\\begin{tabular}{ll|' + 'ccccc|' * (len(new_tasks.values()) - len(baseline_names)) * 2 + 'cc|cc}\n']
    second_line = [data[1]]
    header = list(new_tasks.values())
    header = header[:-2]
    header = ['LLM'] + ['\multicolumn{' + str(2*top_ks[0]) + '}{c}{' + task + '}' for task in header] + ['\multicolumn{2}{c}{' + new_tasks['closedbook'] + '}' , '\multicolumn{2}{c}{' + new_tasks['oracle'] + '}']
    subheader = ['\multicolumn{' + str(top_ks[0]) + '}{c}{Match}' , 
                        '\multicolumn{' + str(top_ks[0]) + '}{c}{Recall}' 
                        ] * (len(new_tasks.values()) - len(baseline_names))
    subheader = subheader + ['Match', 'Recall'] * len(baseline_names)
    subsubheader = [str(i + 1) for i in range(top_ks[0])] * (len(new_tasks.values()) - len(baseline_names)) * 2 
    subsubheader = subsubheader + ['-'] * len(baseline_names) * 2
    # add an '&' in front of every line except the last two lines
    data[4:] = [line if line.startswith('\\') else ' & ' + line for line in data[4:]]

    header_underlines = ['\cmidrule(lr){1-2}'] + ['\cmidrule(lr){'+ str(i) +'-'+ str(i + 9) +'}' for i in range(3, top_ks[0]*6 + 2, 10)] + [str('\cmidrule(lr){' + str(top_ks[0]* 6 + 3) + '-' + str(top_ks[0]* 6 + 4) + '}'), str('\cmidrule(lr){' + str(top_ks[0]* 6 + 5) + '-' + str(top_ks[0]* 6 + 6) + '}')]
    # add the header
    # remove the original header add the third line but keep the first and second line
    # put them below one another and keep the correct spacing!
    data = first_line + second_line + ['&' +' & '.join(header) + '\\\ '] + [' '.join(header_underlines)] + ['\n' + '&  & ' + ' & '.join(subheader) + '\\\ ' + '\n'] + ['\n' +  ' &  & ' + ' & '.join(subsubheader)+ '\\\ ' + '\n'] + [data[3]]+ ['\multirow{3}{*}{\\rotatebox[origin=c]{90}{' + dataset.replace('_', '-') +'}}'] +data[4:]
    print(data)
    with open(f'figures/table_{dataset}.tex', 'w') as file:
        for line in data:
            file.write(line)


if __name__ == '__main__':
    models = ['tinyllamachat', 'llama2_7bchat', 'solar107b', 'mixtral7bchat']
    top_ks = [10]
    dataset = 'kilt_nq'
    #random, relevant_not_correct or relevant
    tasks = ['relevant']
    table(models, top_ks, dataset, tasks)