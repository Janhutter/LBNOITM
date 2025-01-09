import datasets
import os



def load_combine_datasets():
    number_of_proc = os.cpu_count()

    oracle = datasets.load_from_disk('datasets/kilt-100w_full_oracle_provenance') #5141410_129 is last id
    retrieved = datasets.load_from_disk('datasets/kilt-100w_full') # '24853657' is last id

    print(len(oracle))
    print(len(retrieved))
    # # append 'oracle_' to all of the 'id' columns in oracle. use map to apply to all columns
    oracle = oracle.map(lambda x: {'content': x['content'], 'id': 'oracle_' + x['id']}, num_proc=number_of_proc)
    
    retrieved = retrieved.remove_columns('wikipedia_id')

    # combine datasets
    combined = datasets.concatenate_datasets([oracle, retrieved])

    # save combined dataset
    combined.save_to_disk('datasets/kilt-100w_full')

if __name__ == '__main__':
    load_combine_datasets()