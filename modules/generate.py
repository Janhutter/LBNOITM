from torch.utils.data import DataLoader
from tqdm import tqdm
from hydra.utils import instantiate
# Generate
from torch.utils.data import DataLoader
from torch.utils.data import Dataset, DataLoader
import torch
from modules.dataset import Tokenized_Sorted_Dataset
from warnings import warn




class Generate():
    def __init__(self, 
                 prompt=None,
                 init_args=None, 
                 batch_size=1
                 ):

        self.batch_size = batch_size
        # instatiate model

        if 'visualize_attention' not in init_args:
            warn('visualize_attention not found in init_args, setting to False')
            self.visualize_attention = False
        else:
            warn(f'visualize attention only supported for llm models, not vllm models yet.')
            self.visualize_attention = True


        self.model = instantiate(init_args, prompt=prompt)

    def eval(self, dataset):
        tokenized_and_sorted_dataset = Tokenized_Sorted_Dataset(dataset, self.model, training=False)
        if self.visualize_attention:
            dataloader = DataLoader(tokenized_and_sorted_dataset, collate_fn=lambda l: self.model.collate_fn(l, eval=True), num_workers=4)
        else:
            dataloader = DataLoader(tokenized_and_sorted_dataset, batch_size=self.batch_size, collate_fn=lambda l: self.model.collate_fn(l), num_workers=4)


        responses, instructions, query_ids, queries, labels, ranking_labels = list(), list(), list(), list(), list(), list()
        for data_dict in tqdm(dataloader, desc='Generating'):
            id_ = data_dict['q_id']
            instruction = data_dict['instruction']
            query_ids += id_
            label = data_dict['label']
            labels += label
            queries += data_dict['query']
            ranking_labels += data_dict['ranking_label']
            instructions += instruction
            if not self.visualize_attention:
                generated_response = self.model.generate(data_dict['model_input'])
                responses += generated_response
            else:
                attentions = self.model.visualize_attention(data_dict['model_input'], data_dict['instruction'])
                responses += attentions

        return query_ids, queries, instructions, responses, labels, ranking_labels
