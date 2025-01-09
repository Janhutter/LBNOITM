#this script try to run the trained models from the emb folder to evalaute the performance of the model on other dataset
import os
import json
import argparse
# import method for reading yaml files
from omegaconf import OmegaConf
from tqdm import tqdm
from datasets.fingerprint import Hasher

def main(args):
    from modules.rag import RAG
    folder = args.folder
    split = args.split
    target_dataset = args.target_dataset

    sub_folders = [f.path for f in os.scandir(folder) if f.is_dir()]
    sub_folders = [f for f in sub_folders if 'tmp_' not in f]
    for sub_folder in tqdm(sub_folders):
        #first check if train is in the folder
           if os.path.exists(os.path.join(sub_folder, 'train')):
               try:
                    with open('config/rag.yaml', 'r') as f:
                        config = OmegaConf.load(f)

                    print('evaluating', sub_folder)
                    train_folder = os.path.join(sub_folder, 'train')
                    epochs = [f.path for f in os.scandir(train_folder) if f.is_dir()]
                    epochs = [f for f in epochs if 'checkpoint-' in f]
                    epochs = sorted(epochs, key=lambda x: int(x.split('checkpoint-')[-1]))
                    last_epoch = epochs[-1]
                    #get the model name
                    #load config.yaml to the config, but remove the train part
                    config_file = os.path.join(sub_folder, 'config.yaml')
                    with open(config_file, 'r') as f:
                       #use safe_load instead of full_load
                        config_specific = OmegaConf.load(f)
                        config.update(config_specific)

                    #remove the train part
                    config.pop('train')
                    dataset_config_file = os.path.join("config", "dataset", target_dataset + ".yaml")
                    if not os.path.exists(dataset_config_file):
                        print('dataset config file not found')
                        continue
                    with open(dataset_config_file, 'r') as f:
                        dataset_config = OmegaConf.load(f)
                        config.dataset.update(dataset_config)

                    #change model name to the last epoch
                    config.generator.init_args.model_name = last_epoch
                    run_name = f'{Hasher.hash(str(config))}'
                    experiment_folder = os.path.join(folder, run_name)
                    experiment_tmp_folder = os.path.join(folder, 'tmp_' + run_name)
                    if os.path.exists(experiment_folder):
                        if os.path.exists(os.path.join(experiment_folder, "eval_dev_metrics.json")):
                            print('experiment eval file already exists')
                            continue
                    rag = RAG(**config, config=config)
                    rag.eval(dataset_split=split)
                    print('evaluated', sub_folder)
               except Exception as e:
                print('error', e)
                continue



if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--folder', type=str, default="experiments/")
    argparser.add_argument('--split', type=str, default='dev')
    argparser.add_argument('--target_dataset', type=str, default='wizard_of_wikipedia')
    args = argparser.parse_args()

    main(args)