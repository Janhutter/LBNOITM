from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import json

def check_tokens(model, full_modelname, task, top_ks, dataset):
    tokenizer = AutoTokenizer.from_pretrained(full_modelname)

    instruction = "### System:\nYou are a helpful assistant. Your task is to extract relevant information from provided documents and to answer to questions as short as possible.\n\n### User:\nBackground:\nDocument 1: List of Academy Award-winning families. The Hustons were the first three generation family of winners. The others are the Coppolas and, technically, the Farrow/Previn/Allens.\nDocument 2: Acanthocardia echinata. also prominently grooved. The prickly cockle is found in the British Isles and northwestern Europe. It lives within a few centimetres of the sea bottom, at depths of 3 m or more. Dead shells are commonly washed up on the beach.\nDocument 3: SMS Wittelsbach. of the IV Squadron were tasked with coastal defense duties along Germany's North Sea coast against incursions from the British Royal Navy. On 3 April, \"Wittelsbach\" went into drydock in Kiel for periodic maintenance, before conducting training exercises in the western Baltic with the other ships of VII Division of IV Squadron, which included , , and . The German Army requested naval assistance for its campaign against Russia; Prince Heinrich, the commander of all naval forces in the Baltic, made VII Division, IV Scouting Group, and the torpedo boats of the Baltic fleet available for the operation.\nDocument 4: Nuyaka Mission. that the family preferred that the name should be from the Creek language. Therefore, Nuyaka Mission was named for the nearby Creek town of Nuyaka. According to one source, the name Nuyaka is from the Creek pronunciation for New York, which was the site of a meeting between President George Washington and 26 Creek chiefs. The meeting was to discuss a treaty and to obtain a cession of Creek land to the U, S. Government. Reportedly, the Creeks were so impressed with New York City that they named one of their towns for it. White men wrote the\nDocument 5: Theogonia. Theogonia Theogonia may refer to:\n\n\nQuestion: who are the only 2 families that have had 3 generations of oscar winners\n\n### Assistant:\n",

    # instruction_tokenized = tokenizer.encode(instruction)

    # split on \n Document but do keep these inside the strings
    splitted = instruction.split("\nDocument")
    print(splitted)

    # for top_k in top_ks:
    #     inputs = []
    #     with open(f'experiments/{model}_top{top_k}{task}_oracle{0}_{dataset}/eval_dev_out.json') as f:
    #         data = json.load(f)
    #         for row in data:
    #             inputs.append(row['instruction'])

    #     print(inputs[:5])
    #     token_lengths = []
    #     for input in inputs:
    #         tokens = tokenizer.encode(input)
    #         # check the amount of tokens
    #         num_tokens = len(tokens)
    #         token_lengths.append(num_tokens)

    #     max_length = 2048
    #     print(f"Max token length: {max(token_lengths)}")
    #     print(f"Top_ks: {top_k}")
    #     # amount of times that token_lengths is greater than max_length
    #     print(f"Amount of times that token_lengths is greater than max_length: {sum([1 for length in token_lengths if length > max_length])}")
    #     print(f"Amount of times that token_lengths is not greater than max_length: {sum([1 for length in token_lengths if length <= max_length])}")

if __name__ == '__main__':
    model = 'llama2_7bchat'
    full_modelname = "meta-llama/Llama-2-7b-chat-hf"
    task = 'random'
    top_ks = [10]
    dataset = 'kilt_nq'
    check_tokens(model, full_modelname, task, top_ks, dataset)