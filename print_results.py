import argparse
import json
from pathlib import Path
from omegaconf import OmegaConf
import os 
import pandas as pd

def get_info(file):
    res= json.load( open(file))
    nbres=len(res)
    llen=[]
    nbsub=0
    for ex in res:
        llen.append(len(ex['Pred'].split(' ')))
        if ex['SUBSTR']:nbsub+=1 
    
    return nbres,nbsub/nbres,sum(llen)/nbres


def get_em_score(file):
    res= json.load( open(file))
    return res['em']

def get_bem_score(file):
    with open(file) as fd:
        return float(fd.readline().strip())

def get_config(file, split):
    config = OmegaConf.load(file)
    dataset_doc = config['dataset'][split]['doc']['init_args']['_target_'].replace('modules.dataset_processor.', '')
    dataset_query = config['dataset'][split]['query']['init_args']['_target_'].replace('modules.dataset_processor.', '')
    retriever = config['retriever']['init_args']['model_name'] if 'retriever' in config and 'init_args' in config['retriever'] else None
    reranker = config['reranker']['init_args']['model_name'] if 'reranker' in config and 'init_args' in config['reranker'] else None
    generator = config['generator']['init_args']['model_name'] if 'generator' in config and 'init_args' in config['generator'] else None
    prompt = config['prompt'] if 'prompt' in config else ''

    retrieve_top_k = config['retrieve_top_k'] if 'retriever' in config else '-'
    rerank_top_k = config['rerank_top_k'] if 'reranker' in config else '-'
    return dataset_query, dataset_doc, retriever, reranker, generator, prompt, retrieve_top_k, rerank_top_k

def get_scores(file, decimals=2):
    data = json.load(open(file))
    bem = float(data['BEM']) if 'BEM' in data else None
    ll7b = float(data['LLM_ll7b']) if 'LLM_ll7b' in data else None
    ll13b = float(data['LLM_ll13b']) if 'LLM_ll13b' in data else None
    ll70b = float(data['LLM_ll70b']) if 'LLM_ll70bsol' in data else None
    mix7b = float(data['LLM_mix7b']) if 'LLM_mix7b' in data else None
    m = float(data['M'])
    em = float(data['EM'])
    f1 = float(data['F1'])
    precision = float(data['Precision'])
    recall = float(data['Recall'])
    rouge1 = float(data['Rouge-1'])
    rouge2 = float(data['Rouge-2'])
    rougel = float(data['Rouge-L'])

    return m, em, f1, precision, recall, rouge1, rouge2, rougel, bem, ll7b,ll13b,ll70b,  mix7b

def get_generation_time(file):
    data = json.load(open(file))
    return data['Generation time']

def get_ranking_metrics(file):
    data = json.load(open(file))
    return data['P_1']


def main(args):
    
    folder_path = Path(args.folder)
    ltuple=[]
    split = args.split
    for current_folder in folder_path.iterdir():
        skip = False
        try:
            if current_folder.is_dir() and not 'tmp_' in str(current_folder):
                gen_time = None
                ranking_metric = None
                files = [f.name for f in current_folder.iterdir()]

                if f'eval_{split}_metrics.json' in files:

                    for file_in_subfolder in current_folder.iterdir():
                        # try:
                        if 'config.yaml' in str(file_in_subfolder):
                            dataset_query, dataset_doc, retriever, reranker, generator, prompt, retrieve_top_k, rerank_top_k = get_config(file_in_subfolder, split)

                        if f'eval_{split}_metrics.json' in str(file_in_subfolder):
                            m, em, f1, precision, recall, rouge1, rouge2, rougel, bem, ll7b,ll13b,ll70b,  mix7b= get_scores(file_in_subfolder)
                        if f'eval_{split}_generation_time.json' in str(file_in_subfolder) :
                            gen_time = get_generation_time(file_in_subfolder) 

                        if f'eval_{split}_ranking_metrics.json' in str(file_in_subfolder) :
                            ranking_metric = get_ranking_metrics(file_in_subfolder)
                        # except:
                        #     print(f'Failed to load {current_folder}!')       
                    ltuple.append([current_folder, retriever, ranking_metric, reranker, generator, prompt, gen_time, dataset_query, retrieve_top_k, rerank_top_k, m, em, f1, precision, recall, rouge1, rouge2, rougel, bem, mix7b])
                    #ltuple.append([current_folder, retriever, reranker, generator, gen_time, dataset_query, retrieve_top_k, rerank_top_k, m, em, f1, precision, recall, rouge1, rouge2, rougel, bem, ll7b, ll7bs, ll13b, ll13b_short, ll13bswoq, mix7b, mix7bswoq])
        except:
            print(f'Skipping {current_folder} due to parsing errors!')
    
    if len(ltuple) == 0:
        print(f'No results in folder "{args.folder}" yet!')
        exit()
    df= pd.DataFrame(ltuple)
    df.columns = ['exp_folder', 'Retriever', 'P_1', 'Reranker', 'Generator', 'prompt', 'gen_time', 'query_dataset', "r_top", "rr_top", "M", "EM", "F1", "P", "R", "Rg-1", "Rg-2", "Rg-L", "BEM", "Mix7b"]
    #df.columns = ['exp_folder', 'Retriever', 'Reranker', 'Generator', 'gen_time', 'query_dataset', "r_top", "rr_top", "M", "EM", "F1", "P", "R", "Rg-1", "Rg-2", "Rg-L", "BEM", "ll7b", "ll7bs", "ll13b", "ll13bs","ll13bswoq" , "mix7b", "mix7bswoq"]

    df=df.sort_values(by=[args.sort])
    print('Split:', args.split)
    print(df.to_markdown(floatfmt=".2f"))
    if args.csv:
        os.makedirs('results', exist_ok=True)
        file_name = args.folder.replace('/', '_')
        df.to_csv(f'results/{file_name}.csv', index=False)
    



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--folder", type=str, default='experiments')
    parser.add_argument("--split", type=str, default='dev')
    parser.add_argument("--sort", type=str, default="Generator")
    parser.add_argument("--csv", action='store_true')

    args = parser.parse_args()
    main(args)
