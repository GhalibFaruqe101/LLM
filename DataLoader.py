
#%%


with open('LLM/text.txt', 'r', encoding='utf-8') as f:
    text = f.read()
print(len(text))
char = sorted(set(text))
print(char)



print(text[:99])


import re

preprocess_data = re.split(r'([,.:;?_!"()\']|--|\s)' ,text)
preprocess_data = [item.split() for item in preprocess_data if item.strip()]
print(preprocess_data[:50])



print(len(preprocess_data))





words = sorted({token for item in preprocess_data for token in item})

print(len(words))


vocab = { token:integer for integer, token in enumerate(words)}


 



class Tokenizer:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i:s for s,i in vocab.items()}
    
    def encode(self, text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
                                
        preprocessed = [
            item.strip() for item in preprocessed if item.strip()
        ]
        preprocessed =[ item if item in self.str_to_int
                       else "<unk>" for item in preprocessed]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids
        
    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        # Replace spaces before the specified punctuations
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text


# pass the vocabulary mapping (dict) rather than the list of tokens
# 'vocab' is created in a later cell (index 12), so execute that cell before this one
tokenizer = Tokenizer(vocab)


text = """How slowly the time passes here, encompassed as I am by frost and snow!
Yet a second step is taken towards my enterprise. I have hired a
vessel and am occupied in collecting my sailors; those whom I have
already engaged appear to be men on whom I can depend and are certainly
possessed of dauntless courage."""


ids = tokenizer.encode(text)
print(ids[:50])


all_tokens = sorted(list(set(words)))
all_tokens.extend(["<unk>"])
vocab = {token: integer for integer, token in enumerate(all_tokens)}


# BytePair encoding
# 


# %pip install tiktoken



import importlib
import tiktoken


n_text = """How slowly the txt time passes here, encompassed as I am by frost and snow!
Yet a second step is taken towards my enterprise. I have hired a
vessel and am occupied in collecting my sailors; those whom I have
already engaged appear to be men on whom I can depend and are certainly
possessed of dauntless courage."""


tokenizer = tiktoken.get_encoding("gpt2")
print(tokenizer.encode(n_text, allowed_special={"<|unk>|"})[:50])


# Data Sampling using sliding window


# %pip install --upgrade pip



# %pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128


import torch


# with open('LLM/text.txt', 'r', encoding='utf-8') as f:
#     raw_text = f.read()
#     enc_text= tokenizer.encode(text)
#     print(len(enc_text ))


# enc_sample = enc_text[1000:2000]
# x=enc_sample[:4]
# y=enc_sample[1:4+1]
# dec_x= tokenizer.decode(x)
# dec_y= tokenizer.decode(y)

# print(f"x: {dec_x}")
# print(f"y: {dec_y}")


# for i in range(1,4+1):
#     context = enc_sample[:i]
#     desired = enc_sample[i]
#     print(tokenizer.decode(context), '--> ',tokenizer.decode([desired]))


from torch.utils.data import Dataset, DataLoader


# Dataset
# - stride --> controled overlap (between cunks)
# 


class Dataset_V1(Dataset):
    def __init__(self,txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []
#tokenizing the text
        token_ids = tokenizer.encode(txt, allowed_special={ "<|endoftext>"})
        assert len(token_ids) > max_length, "tokenized input equal or +1"
#sliding window
        for i in range(0, len(token_ids)-max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)
    

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


        
   




# dataloader
# - txt --> raw input
# - batch_size --> no of samples
# - drop_batch --> whether to drop last batch if is smaller than batch_size
# - num_workers --> no of subprocessees for data loading


def create_dataloader_v1(txt, batch_size=4, max_length=256, 
                         stride=128, shuffle=True, drop_last=True,
                         num_workers=0):  
    tokenizer = tiktoken.get_encoding("gpt2")

    dataset = Dataset_V1(txt, tokenizer, max_length, stride)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader


with open('LLM/text.txt', 'r', encoding='utf-8') as f:
    raw_text = f.read()


# tiktoken.__version__


dataloader = create_dataloader_v1(
    raw_text, batch_size=8, max_length=4, stride=4, shuffle=False
)

data_iter = iter(dataloader)
input, target = next(data_iter)
print("Inputs:\n", input, input.shape)
print("Targets:\n", target)


# embeding layer
# 


vocab_size = 50247
output_dim =256
token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
token_embedding = token_embedding_layer(input)
context_length = 4
pos_embedding = torch.nn.Embedding(context_length, output_dim)
print(token_embedding.shape)


