from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from transformers import BitsAndBytesConfig

class LLM():
    def __init__(self, model_name):
        quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type='nf4',
                bnb_4bit_compute_dtype='bfloat16',
            )

        self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    quantization_config=quant_config,
                    attn_implementation="eager",
                    torch_dtype=torch.bfloat16,
                    device_map='auto'
                )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_new_tokens = 64
        self.max_doc_len = 2048
        self.quantization = "int4"

        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token = self.tokenizer.bos_token

        self.model = self.model.bfloat16()
        self.model.eval()
        self.model.config.pretraining_tp = 1

    def tokenize_instruction(self, instruction, input_length):
        # Split the instruction into individual sections
        # so we get [instruction, docs1, doc2, ..., question]
        splitted = instruction.split("\nDocument")
        
        for i, doc in enumerate(splitted):
            if i == 0:
                continue
            splitted[i] = "Document" + doc
            splitted[i-1] = splitted[i-1] + "\n"
        
        last_doc, question = splitted[-1].split("Question:")
        question = "Question:" + question
        splitted[-1] = last_doc
        splitted.append(question)

        # Create bins for each section
        bins = []
        start = 0
        for i, doc in enumerate(splitted):
            tokenized_doc = self.tokenizer.encode(doc)
            if i != 0:
                tokenized_doc = tokenized_doc[1:]
            if i != len(splitted) - 1:
                tokenized_doc = tokenized_doc[:-1]
            bins.append((start, start + len(tokenized_doc)))
            start += len(tokenized_doc)

        # replace the end of the last bin with the end of the output tokens
        bins[-1] = (bins[-1][0], input_length + 1)


        return bins

    def get_normalized_attention(self, attentions, bins):
        # take the mean over the sequence length
        document_scores = []
        for start, end in bins:
            doc_doc_attention = attentions[start: start+3]
            doc_attention = attentions[start+3: end]
            # take the mean over the output tokens
            document_scores.append([doc_doc_attention.mean(), doc_attention.mean()])

        # normalizing
        # document_scores = [round(float(i)/sum(document_scores), 4) for i in document_scores]
        return document_scores
    

    def visualize_attention(self, instr_tokenized, instr_untokenized):
        input_ids = instr_tokenized['input_ids'].to("cuda")
        attention_mask = instr_tokenized['attention_mask'].to("cuda")
        output = self.model.generate(
            input_ids,
            attention_mask=attention_mask,
            output_attentions=True,
            do_sample=False,
            max_new_tokens=self.max_new_tokens,
            return_dict_in_generate=True
        )
        # (attention per generated token, layers, batch_size, num_heads, sequence_length, sequence_length)
        # 
        # take the attention while generating the first token
        output_attentions = output['attentions']

        all_layers = []

        bins = self.tokenize_instruction(instr_untokenized[0], input_ids.shape[1])
        output_attentions_list = [
            torch.stack(output_attentions[attn_index], dim=0)[:, :, :, :, :input_ids.shape[1]]
            for attn_index in range(1, len(output_attentions))
        ]

        print('instr_tokenized', instr_tokenized)
        print('instr_untokenized', instr_untokenized)
        # input shape
        print('input_ids.shape:', input_ids.shape)
        # print('untokenized:', instr_untokenized)
        # print('tokenized:', instr_tokenized)
        output_ids = output['sequences']

        prompt_len = instr_tokenized['input_ids'].size(1)
        # print(prompt_len)
        generated_ids = output_ids[:, prompt_len:]

        decoded = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        print('response:', decoded[0])
        attentions = torch.stack(output_attentions_list, dim=0)
        print('attentions.shape:', attentions.shape)

        attentions = attentions.to(torch.float32)
        attentions = attentions.mean(axis=0)
        amount_of_layers = attentions.size(0)
        used_layers = [0, amount_of_layers // 2, amount_of_layers - 1]
        print('used_layers:', used_layers)
        for layer in used_layers:
            all_document_scores = []
            # take the mean over the heads
            layer_attentions = attentions[layer][0, :, :, :]
        

            print('again attentions.shape', layer_attentions.shape)
            # take the mean over the heads
            layer_attentions = layer_attentions.mean(axis=0).squeeze(0).detach().cpu().numpy()

            document_scores = self.get_normalized_attention(layer_attentions, bins)

            all_document_scores.append(document_scores)
            

            # go through all_document_scores and determine the average score for each position
            all_document_scores = torch.tensor(all_document_scores)
            all_document_scores = all_document_scores.mean(axis=0).tolist()
            all_layers.append(all_document_scores)
            print('all_document_scores:', all_document_scores)
            print('doc position with max value:', all_document_scores[1:-1].index(max(all_document_scores[1:-1])))

        return str([decoded[0], all_layers])


model = LLM("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
# model = LLM("meta-llama/Llama-2-7b-chat-hf")

# pos 4
instr_untokenized_4 = ['<|system|>\nYou are a helpful assistant. Your task is to extract relevant information from provided documents and to answer to questions as short as possible.</s>\n<|user|>\nBackground:\nDocument 1: Andrew Bridge (lawyer). California Regents\' Lecturer. In 2015, he was appointed Commissioner to the Los Angeles County Probation Commission. He is a founding Director of The New Village Charter School for Girls, a charter school for pregnant and parenting teens and California\'s first charter school solely for girls. Section:External links.\nDocument 2: Villa Fiorita, Brugherio. its park setting and its optimal exposure to sunlight. Section:See also.\nDocument 3: USS Greenling (SS-213). the Marshall Islands and Caroline Islands. The submarine attacked the cargo ship "Seia Maru" four times 30 April – 1 May off Eniwetok, but due to faulty torpedoes was not able to sink her. The tenacious submarine even closed for a night gunfire attack in an attempt to cripple her adversary. Finally forced by Japanese aircraft to break off the attack, "Greenling" turned her attention to the huge Japanese base at Truk. As the Japanese converged on the Solomon Islands, Truk became a busy shipping point and a fertile ground for submarine operations. The submarine recorded her\nDocument 4: Great Plague of London. The Great Plague, lasting from 1665 to 1666, was the last major epidemic of the bubonic plague to occur in England. It happened within the centuries-long time period of the Second Pandemic, an extended period of intermittent bubonic plague epidemics which originated in China in 1331, the first year of the Black Death, an outbreak which included other forms such as pneumonic plague, and lasted until 1750.\nDocument 5: Yoel Goldman. is one of the most prominent developers in Brooklyn credited with helping to gentrify Williamsburg, Bushwick, Greenpoint, Borough Park, and Bedford-Stuyvesant. Section:Developments.\n\n\nQuestion: when did the great plague of london start and end</s>\n<|assistant|>\n\nResponse:\n']
instr_tokenized_4 = {'input_ids': torch.tensor([[    1,   529, 29989,  5205, 29989, 29958,    13,  3492,   526,   263,
          8444, 20255, 29889,  3575,  3414,   338,   304,  6597,  8018,  2472,
           515,  4944, 10701,   322,   304,  1234,   304,  5155,   408,  3273,
           408,  1950, 29889,     2, 29871,    13, 29966, 29989,  1792, 29989,
         29958,    13, 10581, 29901,    13,  6268, 29871, 29896, 29901, 11571,
         16230,   313, 10653,  7598,   467,  8046,  2169,  1237, 29915,   365,
           522,  9945, 29889,   512, 29871, 29906, 29900, 29896, 29945, 29892,
           540,   471, 10658, 11444,   261,   304,   278,  4602, 10722,  5127,
          1019, 29890,   362, 11444, 29889,   940,   338,   263,  1476,   292,
         15944,   310,   450,  1570, 24720,  2896,   357,  4523,   363, 23992,
         29892,   263,  1373,   357,  3762,   363,   758,  5138,   424,   322,
          3847,   292,   734,   575,   322,  8046, 29915, 29879,   937,  1373,
           357,  3762, 14419,   368,   363, 14000, 29889,  9779, 29901, 25865,
          2988, 29889,    13,  6268, 29871, 29906, 29901, 14439,   383,  1611,
          2028, 29892,  8135, 29887,  2276,   601, 29889,   967, 14089,  4444,
           322,   967, 14413, 14060,   545,   304,  6575,  4366, 29889,  9779,
         29901, 13393,   884, 29889,    13,  6268, 29871, 29941, 29901, 17676,
          7646,  1847,   313,  1799, 29899, 29906, 29896, 29941,   467,   278,
         23072, 17839,   322, 26980, 17839, 29889,   450,  1014, 24556, 22630,
           278, 17040,  7751,   376,  2008,   423,  1085, 29884, 29908,  3023,
          3064, 29871, 29941, 29900,  3786,   785, 29871, 29896,  2610,  1283,
          1174,  9429,   300,   554, 29892,   541,  2861,   304, 12570, 29891,
          4842,  9795, 29877,   267,   471,   451,  2221,   304, 28169,   902,
         29889,   450,   260,  2386,  8802,  1014, 24556,  1584,  5764,   363,
           263,  4646, 13736,  8696,  5337,   297,   385,  4218,   304, 14783,
           407,   280,   902, 19901,   653, 29889,  9788, 11826,   491, 10369,
         15780,   304,  2867,  1283,   278,  5337, 29892,   376, 24599,  1847,
         29908,  6077,   902,  8570,   304,   278, 12176, 10369,  2967,   472,
         17238, 29895, 29889,  1094,   278, 10369,  5486,  3192,   373,   278,
          4956, 18192, 17839, 29892, 17238, 29895,  3897,   263, 19587,   528,
         17347,  1298,   322,   263, 19965,   488,  5962,   363,  1014, 24556,
          6931, 29889,   450,  1014, 24556, 10478,   902,    13,  6268, 29871,
         29946, 29901,  7027,  1858,  3437,   310,  4517, 29889,   450,  7027,
          1858,  3437, 29892,  1833,   292,   515, 29871, 29896, 29953, 29953,
         29945,   304, 29871, 29896, 29953, 29953, 29953, 29892,   471,   278,
          1833,  4655,  9358,   680, 13076,   310,   278,   289,   431,  8927,
           715,  3437,   304,  6403,   297,  5408, 29889,   739,  9559,  2629,
           278, 21726, 29899,  5426,   931,  3785,   310,   278,  6440,  6518,
         24552, 29892,   385, 10410,  3785,   310,  1006, 18344,   296,   289,
           431,  8927,   715,  3437,  9358,   680, 29885,  1199,   607,  3978,
           630,   297,  7551,   297, 29871, 29896, 29941, 29941, 29896, 29892,
           278,   937,  1629,   310,   278,  6054, 14450, 29892,   385,   714,
          8690,   607,  5134,   916,  7190,  1316,   408,   282, 29765,  8927,
           715,  3437, 29892,   322,  1833,   287,  2745, 29871, 29896, 29955,
         29945, 29900, 29889,    13,  6268, 29871, 29945, 29901,   612, 29877,
           295,  6650,  1171, 29889,   338,   697,   310,   278,  1556, 19555,
         18777,   297, 18737, 13493,  6625,  1573,   411, 19912,   304,  8116,
         29878,  1598, 11648,  3074, 29892, 24715,  6669, 29892,  7646,  3149,
         29892,  6780,   820,  4815, 29892,   322, 14195,  4006, 29899,   855,
          8631,  1960,   424, 29889,  9779, 29901, 21956,  1860, 29889,    13,
            13,    13, 16492, 29901,   746,  1258,   278,  2107,   715,  3437,
           310,   301,   898,   265,  1369,   322,  1095,     2, 29871,    13,
         29966, 29989,   465, 22137, 29989, 29958,    13,    13,  5103, 29901,
            13]]), 'attention_mask': torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1]])}

# pos 0
instr_untokenized_0= ['<|system|>\nYou are a helpful assistant. Your task is to extract relevant information from provided documents and to answer to questions as short as possible.</s>\n<|user|>\nBackground:\nDocument 1: Great Plague of London. The Great Plague, lasting from 1665 to 1666, was the last major epidemic of the bubonic plague to occur in England. It happened within the centuries-long time period of the Second Pandemic, an extended period of intermittent bubonic plague epidemics which originated in China in 1331, the first year of the Black Death, an outbreak which included other forms such as pneumonic plague, and lasted until 1750.\nDocument 2: Andrew Bridge (lawyer). California Regents\' Lecturer. In 2015, he was appointed Commissioner to the Los Angeles County Probation Commission. He is a founding Director of The New Village Charter School for Girls, a charter school for pregnant and parenting teens and California\'s first charter school solely for girls. Section:External links.\nDocument 3: Villa Fiorita, Brugherio. its park setting and its optimal exposure to sunlight. Section:See also.\nDocument 4: USS Greenling (SS-213). the Marshall Islands and Caroline Islands. The submarine attacked the cargo ship "Seia Maru" four times 30 April – 1 May off Eniwetok, but due to faulty torpedoes was not able to sink her. The tenacious submarine even closed for a night gunfire attack in an attempt to cripple her adversary. Finally forced by Japanese aircraft to break off the attack, "Greenling" turned her attention to the huge Japanese base at Truk. As the Japanese converged on the Solomon Islands, Truk became a busy shipping point and a fertile ground for submarine operations. The submarine recorded her\nDocument 5: Yoel Goldman. is one of the most prominent developers in Brooklyn credited with helping to gentrify Williamsburg, Bushwick, Greenpoint, Borough Park, and Bedford-Stuyvesant. Section:Developments.\n\n\nQuestion: when did the great plague of london start and end</s>\n<|assistant|>\n\nResponse:\n']
instr_tokenized_0 = {'input_ids': torch.tensor([[    1,   529, 29989,  5205, 29989, 29958,    13,  3492,   526,   263,
          8444, 20255, 29889,  3575,  3414,   338,   304,  6597,  8018,  2472,
           515,  4944, 10701,   322,   304,  1234,   304,  5155,   408,  3273,
           408,  1950, 29889,     2, 29871,    13, 29966, 29989,  1792, 29989,
         29958,    13, 10581, 29901,    13,  6268, 29871, 29896, 29901,  7027,
          1858,  3437,   310,  4517, 29889,   450,  7027,  1858,  3437, 29892,
          1833,   292,   515, 29871, 29896, 29953, 29953, 29945,   304, 29871,
         29896, 29953, 29953, 29953, 29892,   471,   278,  1833,  4655,  9358,
           680, 13076,   310,   278,   289,   431,  8927,   715,  3437,   304,
          6403,   297,  5408, 29889,   739,  9559,  2629,   278, 21726, 29899,
          5426,   931,  3785,   310,   278,  6440,  6518, 24552, 29892,   385,
         10410,  3785,   310,  1006, 18344,   296,   289,   431,  8927,   715,
          3437,  9358,   680, 29885,  1199,   607,  3978,   630,   297,  7551,
           297, 29871, 29896, 29941, 29941, 29896, 29892,   278,   937,  1629,
           310,   278,  6054, 14450, 29892,   385,   714,  8690,   607,  5134,
           916,  7190,  1316,   408,   282, 29765,  8927,   715,  3437, 29892,
           322,  1833,   287,  2745, 29871, 29896, 29955, 29945, 29900, 29889,
            13,  6268, 29871, 29906, 29901, 11571, 16230,   313, 10653,  7598,
           467,  8046,  2169,  1237, 29915,   365,   522,  9945, 29889,   512,
         29871, 29906, 29900, 29896, 29945, 29892,   540,   471, 10658, 11444,
           261,   304,   278,  4602, 10722,  5127,  1019, 29890,   362, 11444,
         29889,   940,   338,   263,  1476,   292, 15944,   310,   450,  1570,
         24720,  2896,   357,  4523,   363, 23992, 29892,   263,  1373,   357,
          3762,   363,   758,  5138,   424,   322,  3847,   292,   734,   575,
           322,  8046, 29915, 29879,   937,  1373,   357,  3762, 14419,   368,
           363, 14000, 29889,  9779, 29901, 25865,  2988, 29889,    13,  6268,
         29871, 29941, 29901, 14439,   383,  1611,  2028, 29892,  8135, 29887,
          2276,   601, 29889,   967, 14089,  4444,   322,   967, 14413, 14060,
           545,   304,  6575,  4366, 29889,  9779, 29901, 13393,   884, 29889,
            13,  6268, 29871, 29946, 29901, 17676,  7646,  1847,   313,  1799,
         29899, 29906, 29896, 29941,   467,   278, 23072, 17839,   322, 26980,
         17839, 29889,   450,  1014, 24556, 22630,   278, 17040,  7751,   376,
          2008,   423,  1085, 29884, 29908,  3023,  3064, 29871, 29941, 29900,
          3786,   785, 29871, 29896,  2610,  1283,  1174,  9429,   300,   554,
         29892,   541,  2861,   304, 12570, 29891,  4842,  9795, 29877,   267,
           471,   451,  2221,   304, 28169,   902, 29889,   450,   260,  2386,
          8802,  1014, 24556,  1584,  5764,   363,   263,  4646, 13736,  8696,
          5337,   297,   385,  4218,   304, 14783,   407,   280,   902, 19901,
           653, 29889,  9788, 11826,   491, 10369, 15780,   304,  2867,  1283,
           278,  5337, 29892,   376, 24599,  1847, 29908,  6077,   902,  8570,
           304,   278, 12176, 10369,  2967,   472, 17238, 29895, 29889,  1094,
           278, 10369,  5486,  3192,   373,   278,  4956, 18192, 17839, 29892,
         17238, 29895,  3897,   263, 19587,   528, 17347,  1298,   322,   263,
         19965,   488,  5962,   363,  1014, 24556,  6931, 29889,   450,  1014,
         24556, 10478,   902,    13,  6268, 29871, 29945, 29901,   612, 29877,
           295,  6650,  1171, 29889,   338,   697,   310,   278,  1556, 19555,
         18777,   297, 18737, 13493,  6625,  1573,   411, 19912,   304,  8116,
         29878,  1598, 11648,  3074, 29892, 24715,  6669, 29892,  7646,  3149,
         29892,  6780,   820,  4815, 29892,   322, 14195,  4006, 29899,   855,
          8631,  1960,   424, 29889,  9779, 29901, 21956,  1860, 29889,    13,
            13,    13, 16492, 29901,   746,  1258,   278,  2107,   715,  3437,
           310,   301,   898,   265,  1369,   322,  1095,     2, 29871,    13,
         29966, 29989,   465, 22137, 29989, 29958,    13,    13,  5103, 29901,
            13]]), 'attention_mask': torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1]])}

print('-------------- POS 0 --------------')
model.visualize_attention(instr_tokenized_0, instr_untokenized_0)
print('\n\n\n\n')
print('-------------- POS 4 --------------')
model.visualize_attention(instr_tokenized_4, instr_untokenized_4)