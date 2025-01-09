import pandas as pd
import numpy as np
import datasets
from tqdm import tqdm
import re
from concurrent.futures import ThreadPoolExecutor
import os
import swifter
import nltk
import json
import ast

# make sure to print all columns of pandas 
pd.set_option('display.max_columns', None)

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def lost_in_middle():

    def trec_to_dataframe(filename):
        threenf_trec_data_dict = {'q_id': [], 'd_id': [], 'score': []}
        for l in tqdm(open(filename), desc=f'Loading existing trec run {filename}'):
            q_id, _, d_id, _, score, _ = l.split('\t')
            threenf_trec_data_dict['q_id'].append(q_id)
            threenf_trec_data_dict['d_id'].append(d_id)
            threenf_trec_data_dict['score'].append(float(score))
        threenf_trec_data = pd.DataFrame(threenf_trec_data_dict)
        return threenf_trec_data

    def take_top_oracle(oracle_q):
        # take            import nltk only the first entry with score 100
        oracle_q = oracle_q[oracle_q['score'] == 100]
        # make sure there are no duplicate q_id
        oracle_q = oracle_q.drop_duplicates(subset='q_id', keep='first')
        # append 'oracle_' to every d_id
        oracle_q['d_id'] = 'oracle_' + oracle_q['d_id']
        return oracle_q

    def take_topk_entries(dataframe, topk):
        """
        Take only the top k entries for each unique q_id
        """
        
        dataframe = dataframe.groupby('q_id').head(topk)
        return dataframe

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

    def remove_correct_doc(retrieved, n):
        # use match_criteria to exclude docs from retrieved. Labels = retrieved['content'] and entry = retrieved['doc']
        # use the map function to speed up the proces and you can cut of for a certain question_id if it already has n amount of docs

        # change the type of x['label'] from np.ndarray to list
        retrieved['label'] = retrieved['label'].apply(lambda x: x.tolist() if isinstance(x, np.ndarray) else x)
        retrieved['remove'] = retrieved.swifter.apply(lambda x: not match_criteria(x['label'], x['doc']), axis=1)
        retrieved = retrieved[retrieved['remove']].groupby('q_id').head(n)
        return retrieved


    def get_answer_labels(dataset_name):
        dataset = datasets.load_from_disk(dataset_name)
        answer_labels = dataset.to_pandas()
        return answer_labels


    def dataframe_to_trec(data, filename):
        with open(filename, 'w') as f:
            for q_id in data['q_id'].unique():
                data_q = data[data['q_id'] == q_id]
                length = len(data_q['d_id'].values)
                qs = [q_id] * length
                for rank, (q, d, s) in enumerate(zip(qs, data_q['d_id'].values, data_q['score'].values)):
                    f.write(f'{q}\tq0\t{d}\t{int(rank+1)}\t{s}\trun\n')


    def real_random_doc(retrieved, n, range=24853657):
        """
        Give random d_ids for a given q_id in range and return n amount of them
        """
        new_retrieved = pd.DataFrame(columns=['q_id', 'd_id', 'score'])
        for q_id in retrieved['q_id'].unique():
            d_ids = np.random.randint(0, range, n)
            new_retrieved = pd.concat([new_retrieved, pd.DataFrame({'q_id': [q_id] * n, 'd_id': d_ids, 'score': [0] * n})])

        new_retrieved.reset_index(drop=True, inplace=True)
        return new_retrieved


    def retrieve_random_doc(retrieved, n):
        """
        From retrieved, retrieve n random docs per question
        Returns n amount of docs with score
        """

        new_retrieved = pd.DataFrame(columns=['q_id', 'd_id', 'score'])
        for q_id in retrieved['q_id'].unique():
            retrieved_q = retrieved.sample(n=n)
            retrieved_q['q_id'] = q_id
            retrieved_q['score'] = 0
            new_retrieved = pd.concat([new_retrieved, retrieved_q])

        new_retrieved.reset_index(drop=True, inplace=True)
        return new_retrieved


    def insert_oracle_at_position_k(retrieved, oracle, k):
        """
        insert oracle doc at position k. 0 is first in context. 4 is at the end of top 5.
        """
        count = 0
        new_retrieved = pd.DataFrame(columns=['q_id', 'd_id', 'score'])
        for q_id in retrieved['q_id'].unique():
            retrieved_q = retrieved[retrieved['q_id'] == q_id]
            oracle_q = oracle[oracle['q_id'] == q_id]

            if oracle_q.empty:
                count += 1

            if k > len(retrieved_q):
                raise ValueError('k is larger than number of retrieved docs')

            # oracle at position k of retrieved docs\
            new_entries = pd.DataFrame(columns=['q_id', 'd_id', 'score'], )
            for i in range(len(retrieved_q) + 1):

                if i == k:
                    new_entries = pd.concat([new_entries, oracle_q])

                new_entries = pd.concat([new_entries, retrieved_q.iloc[i:i+1]])

            new_retrieved = pd.concat([new_retrieved, new_entries])

        new_retrieved.reset_index(drop=True, inplace=True)

        print(f'Amount of q_id without oracle: {count}')
        return new_retrieved


    def random_doc_runs(top_k=5, dataset='kilt_nq', full_dataset='kilt-100w_full'):
        """
        Make trec files for random documents per question
        Using filename as a trec file for all of the different question ids
        Before running make sure that top_k is the desired value
        """
        filename = f'runs/run.retrieve.top_1000.{dataset}.{full_dataset[:-5]}.dev.bm25.trec'
        trec_data = trec_to_dataframe(filename)
        trec_data = real_random_doc(trec_data, top_k-1)
        # amount of unique q_id in trec_data
        original_length = len(trec_data['q_id'].unique())

        filename_oracle_1 = f'runs/run.oracle.{dataset}.dev.trec'
        trec_data_oracle = trec_to_dataframe(filename_oracle_1)
        trec_data_oracle = take_top_oracle(trec_data_oracle)

        # only retain entries where score is max, so only 1 entry per unique q_id
        # check if there are 5 or more documents for a question
        # print amount of q_id with less than top_k-1 documents
   

        for i in range(top_k):
            retrieved = insert_oracle_at_position_k(trec_data, trec_data_oracle, i)
            # drop all entries with less than 5 documents
            retrieved = retrieved.groupby('q_id').filter(lambda x: len(x) >= top_k)
            dataframe_to_trec(retrieved, f'runs/run.{dataset}_top{top_k}random_oracle_at_{i}.trec')

        new_length = len(retrieved['q_id'].unique())
        print(f'Random doc run:\nOriginal length: {original_length}, New length: {new_length}')

    # random_doc_runs()


    def relevant_not_correct_doc_runs(top_k=5, dataset='kilt_nq', full_dataset='kilt-100w_full'):
        """
        Make trec files for relevant documents but not correct for each question
        We use top_n run, reverse the order and remove the correct documents.
        Then we take the first top_k-1 documents and insert oracle at desired position.
        Using filename as a trec file for all of the different question ids
        Before running make sure that top_k is the desired value
        """
        if os.path.exists(f'runs/{dataset}_bm25relevant_not_correct.parquet'):
            trec_data = pd.read_parquet(f'runs/{dataset}_bm25relevant_not_correct.parquet')
        else:
            filename = f'runs/run.retrieve.top_1000.{dataset}.{full_dataset[:-5]}.dev.bm25.trec'
            trec_data = trec_to_dataframe(filename)

            # answer labels for the question_ids
            answer_labels = get_answer_labels(f'datasets/{dataset}_validation')

            # reverse the order of the retrieved documents
            # trec_data = trec_data.iloc[::-1]
            trec_data.reset_index(drop=True, inplace=True)

            # oracle docs to combine with relevant but not correct documents
            
            # taking from this document the contents of every d_id that is in the ranking of trec_data
            oracle_docs = datasets.load_from_disk(f'datasets/{full_dataset}_oracle_provenance')
            oracle_docs = oracle_docs.to_pandas()
            # merge the oracle docs with the trec_data on id of oracle_docs and d_id of trec_data
            trec_data = pd.merge(trec_data, oracle_docs, left_on='d_id', right_on='id', how='left')
            trec_data.drop(columns=['id'], inplace=True)

            trec_data.rename(columns={'content': 'doc'}, inplace=True)

            # add answer_labels for every question to trec_data
            trec_data = pd.merge(trec_data, answer_labels, left_on='q_id', right_on='id', how='left')
            trec_data.drop(columns=['id'], inplace=True)
            trec_data.to_parquet(f'runs/{dataset}_bm25relevant_not_correct.parquet')

        # print amount of rows where nan value
        print('NaN entries: ', trec_data.isna().sum())

        original_length = len(trec_data['q_id'].unique())

        filename_oracle_1 = f'runs/run.oracle.{dataset}.dev.trec'
        trec_data_oracle = trec_to_dataframe(filename_oracle_1)
        trec_data_oracle = take_top_oracle(trec_data_oracle)

        trec_data = remove_correct_doc(trec_data, top_k-1)

        # remove the entries with q_id: 
        # entries = ['502101','502172','502630','502665','502683','502714','502873','502894','145507','145538','240079','240093','2400980','2733567']
        # trec_data = trec_data[~trec_data['q_id'].isin(entries)]

        # remove all entries with less than top_k-1 documents
        trec_data = trec_data.groupby('q_id').filter(lambda x: len(x) >= top_k-1)

        for i in range(top_k):
            retrieved = insert_oracle_at_position_k(trec_data, trec_data_oracle, i)
            # drop all entries with less than 5 documents
            retrieved = retrieved.groupby('q_id').filter(lambda x: len(x) >= top_k)
            dataframe_to_trec(retrieved, f'runs/run.{dataset}_top{top_k}relevant_not_correct_oracle_at_{i}.trec')

        new_length = len(retrieved['q_id'].unique())
        print(f'Relevant incorrect doc run:\nOriginal length: {original_length}, New length: {new_length}')


    # relevant_not_correct_doc_runs(5, 'kilt_nq', 'kilt-100w_full')
    # relevant_not_correct_doc_runs(10, 'kilt_nq', 'kilt-100w_full')
    # relevant_not_correct_doc_runs(15, 'kilt_nq', 'kilt-100w_full')


    def relevant_doc_runs(top_k=5, dataset='kilt_nq', full_dataset='kilt-100w_full'):
        """
        Make trec files for top_k bm25 trec runs with oracle at every position
        We use top 50 run of bm25 and insert oracle at desired position.
        Before running make sure that top_k is the desired value
        """

        filename = f'runs/run.retrieve.top_1000.{dataset}.kilt-100w.dev.bm25.trec'
        trec_data = trec_to_dataframe(filename)

        trec_data = take_topk_entries(trec_data, top_k-1)

        filename_oracle_1 = f'runs/run.oracle.{dataset}.dev.trec'
        trec_data_oracle = trec_to_dataframe(filename_oracle_1)
        trec_data_oracle = take_top_oracle(trec_data_oracle)

        original_length = len(trec_data['q_id'].unique())

        for i in range(top_k):
            retrieved = insert_oracle_at_position_k(trec_data, trec_data_oracle, i)
            # drop all entries with less than 5 documents
            retrieved = retrieved.groupby('q_id').filter(lambda x: len(x) >= top_k)
            dataframe_to_trec(retrieved, f'runs/run.{dataset}_top{top_k}relevant_oracle_at_{i}.trec')

        new_length = len(retrieved['q_id'].unique())
        print(f'Relevant doc run: \nOriginal length: {original_length}, New length: {new_length}')

    def change_oracle_file(trec_filename):
        """
        Change the name of oracle doc ids so that they correspond to the doc ids in the new dataset
        So oracle ids are oracle_10593264_2 instead of just 10593264_2
        """
        trec_data = trec_to_dataframe(trec_filename)
        trec_data['d_id'] = 'oracle_' + trec_data['d_id']
        # remove .trec from filename
        trec_filename = trec_filename[:-5]
        dataframe_to_trec(trec_data, f'{trec_filename}_new.trec')

    # do this for every dataset
    # change_oracle_file('runs/run.oracle.kilt_nq.dev.trec')

    def load_correct_trecs():
        """
        Make sure that the trec files have correct amount of documents per question
        """
        task = 'relevant'
        top_ks = [5, 10]
        datasets = ['kilt_nq']
        for dataset in datasets:
            for top_k in top_ks:
                for position in range(top_k):
                    filename = f'runs/run.{dataset}_top{top_k}{task}_oracle_at_{position}.trec'
                    trec_data = trec_to_dataframe(filename)
                    # make sure questions have top_k amount of docs
                    trec_data = trec_data.groupby('q_id').filter(lambda x: len(x) == top_k)
                    dataframe_to_trec(trec_data, f'runs/run.{dataset}_top{top_k}{task}_oracle_at_{position}.trec')

    # load_correct_trecs()

    def trecs_for_lost_in_middle(topks = None, datasets = None, full_datasets = None):
        """
        Make all of the needed trec files for the 3 tasks and datasets!
        Make sure to have bm25 run with top k of 1000 for the datasets
        and a combined dataset of full with oracle provenance docs that has the same
        name as the normal oracle provenance dataset.
        """
        if not topks or not datasets or not full_datasets:
            print('Using default datasets and topks')
            # datasets = ['kilt_nq']
            datasets= ['kilt_hotpotqa']
            full_datasets = ['kilt-100w_full']
            topks = [5, 10]


        for dataset, full_dataset in zip(datasets, full_datasets):

            for top_k in topks:
                random_doc_runs(top_k, dataset, full_dataset)
                relevant_not_correct_doc_runs(top_k, dataset, full_dataset)
                relevant_doc_runs(top_k, dataset, full_dataset)

            change_oracle_file(f'runs/run.oracle.{dataset}.dev.trec')

    # trecs_for_lost_in_middle()

    def position_oracle_in_entry(question):
        """
        Find the position of the oracle inside the question pandas dataframe
        document id starts with oracle_
        """
        # find the index inside of the question dataframe
        new_dataframe = question.reset_index(drop=True)
        oracle_position = new_dataframe[new_dataframe['d_id'].str.contains('oracle')].index[0]
        return oracle_position

    # test = pd.DataFrame({'q_id': [1, 1, 1, 1, 1], 'd_id': ['123123', 'doc2', 'doc3', 'oracle_32178', 'doc4'], 'score': [1, 2, 3, 100, 4]})
    # print(position_oracle_in_entry(test))

    def check_reranker():
        """
        check position of oracle after reranking
        """
        dataset='kilt_nq'
        full_dataset='kilt-100w'
        tasks = ['random', 'relevant_not_correct', 'relevant']
        top_ks = [5, 10]

        data = {'task': [],  'top_k': [], 'original_position': [], 'updated_oracle_pos': []}

        if not os.path.exists('figures/reranker_oracle_positions.csv'):
            for task in tasks:
                for top_k in top_ks:
                    for position in range(top_k):
                        filename = f'experiments/reranking_top{top_k}{task}_oracle{position}_{dataset}/run.rerank.retriever.top_{top_k}.oracle_provenance.rerank.top_{top_k}.{dataset}.{full_dataset}_oracle_provenance.dev.naver_trecdl22-crossencoder-debertav3.pos_{position}.trec'
                        trec_data = trec_to_dataframe(filename)

                        for q_id in trec_data['q_id'].unique():
                            question = trec_data[trec_data['q_id'] == q_id]
                            oracle_position = position_oracle_in_entry(question)
                            data['task'].append(task)
                            data['top_k'].append(top_k)
                            data['original_position'].append(position)
                            data['updated_oracle_pos'].append(oracle_position)


            # make a dataframe of the data
            df = pd.DataFrame(data)
            df.to_csv('figures/reranker_oracle_positions.csv', index=False)
        else:
            df = pd.read_csv('figures/reranker_oracle_positions.csv')

        # for every task and every top_k and also total:
        # amount of times the updated oracle position is not 0
        # amount of times the updated oracle position is not equal to the original position

        for task in tasks:
            for top_k in top_ks:
                for position in range(top_k):
                    updated_oracle_pos = df[(df['task'] == task) & (df['top_k'] == top_k) & (df['original_position'] == position)]['updated_oracle_pos']
                    not_zero = len(updated_oracle_pos[updated_oracle_pos != 0])
                    not_equal = len(updated_oracle_pos[updated_oracle_pos != position])
                    print(f'Task: {task}, Top_k: {top_k}, Position: {position}, Not zero: {not_zero}, Not equal: {not_equal}, Length: {len(updated_oracle_pos)}, percentage: {round(not_zero/len(updated_oracle_pos), 2)}')
    # check_reranker()



    def info_datasets():
        full_name = 'kilt_hotpotqa_validation'
        data = datasets.load_from_disk(f'datasets/{full_name}').to_pandas()
        
        # Initialize tokenizer
        tokenizer = nltk.RegexpTokenizer(r'\w+')
        
        # Tokenize and process the data
        word_lengths = []
        average_summed = []

        for content in tqdm(data['content']):
            words = tokenizer.tokenize(content.lower())
            word_lengths.extend([len(word) for word in words])
            average_summed.append(len(words))
        
        # Calculating averages
        avg_words_per_doc = sum(average_summed) / len(average_summed)
        avg_letters_per_word = sum(word_lengths) / len(word_lengths)
        
        # Printing results
        print(f'Average amount of words per document: {avg_words_per_doc}')
        print(f'Length: {len(average_summed)}')
        print(f'Questions: {len(data["content"])}')
        print(f'Average amount of letters per word: {avg_letters_per_word}')

    info_datasets()

    def check_trecs():
        tasks = ['random', 'relevant_not_correct', 'relevant']
        dataset_name = 'kilt_hotpotqa'
        top_ks = [5, 10]
        for task in tasks:
            for top_k in top_ks:
                filename = f'runs/run.{dataset_name}_top{top_k}{task}_oracle_at_{0}.trec'
                trec_data = trec_to_dataframe(filename)
                print(f'Amount of unique q_id in {filename}: {len(trec_data["q_id"].unique())} in dataset {dataset_name} and top k {top_k} and task {task}')

    # check_trecs()

    def white_space_fix(entry):
        return ' '.join(entry.split(' '))
    
    def cannot_answer(response):
        response = white_space_fix(response)
        return response.startswith('I cannot answer this question') or response.startswith('The provided documents do not') or response.startswith('There is no information') or response.startswith("There's no information") or response.startswith("I cannot") or response.startswith('The provided information does not')

    def check_answers():
        tasks = ['random']
        dataset_name = 'kilt_hotpotqa'
        model = 'llama2_7bchat'
        top_ks = [5, 10]
        for task in tasks:
            for top_k in top_ks:
                filename = f'experiments/{model}_top{top_k}{task}_oracle0_{dataset_name}/eval_dev_out.json'
                with open(filename) as f:
                    data = json.load(f)
                responses = [entry['response'] for entry in data]
                count = sum([1 for response in responses if cannot_answer(response)])
                print(f'Amount of cannot answer responses in {filename}: {count} over total {len(responses)} for task {task} and dataset {dataset_name} and top k {top_k}')

    # check_answers()

if __name__ == '__main__':
    lost_in_middle()