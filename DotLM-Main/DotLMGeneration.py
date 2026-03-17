import torch
import torch.nn as nn
from torch.nn import functional as F
from datetime import datetime
import os
import json
from DotLM import DotLM
import DotLM as DotFunc

# I need to make it so that there are generations (ex. how many messages generated)
def generate_10x():
    # Change this to be general and not just quotes
    print("Quote Generation:")
    model_path = "inspoquotes_model.pth"

    if os.path.exists(model_path):
        #
        vocab_size = 82 #CHANGE THIS TO THE ACTUAL VOCAB SIZE
        block_size = 128
        # batch_size = 8
        #
        model = DotLM(vocab_size, block_size=block_size)
        # optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
        #
        vocab = json.load(open("inspoVocab.json"))
        stoi, itos = vocab["stoi"], vocab["itos"]

        # print("itos:", itos, ", itos type:", type(itos), "\n49th?", itos["49"]) # fixed the keyerror finally

        # encode = lambda s: [stoi[c] for c in s]
        decode = lambda l: ''.join([itos[str(i)] for i in l])
        #
        print("Loading le model.")
        model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=True))

        print("Quotes (x10):")
        
        for i in range(10):
            print(DotFunc.generate(model, stoi, decode, start="Quote: ", length=300, temperature=0.8, top_k=50))

        print("End of x10 quotes")

    else:
        print("No model found, ending prompting.")

def generate(prompt, model_file, vocab_size, vocab_file):
    model_path = model_file + ".pth"

    if os.path.exists(model_path) and os.path.exists(vocab_file):
        vocab = json.load(open(vocab_file))
        stoi, itos = vocab["stoi"], vocab["itos"]

        #
        vocab_size = vocab["vsize"] #CHANGE THIS TO THE ACTUAL VOCAB SIZE
        block_size = 128
        # batch_size = 8
        #
        model = DotLM(vocab_size, block_size=block_size)
        # optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
        #
        

        # print("itos:", itos, ", itos type:", type(itos), "\n49th?", itos["49"]) # fixed the keyerror finally

        # encode = lambda s: [stoi[c] for c in s]
        decode = lambda l: ''.join([itos[str(i)] for i in l])
        #
        #print("Loading le model.")
        model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=True))

        prompt = prompt + "\nAI: "
        
        response = DotFunc.generate(model, stoi, decode, start=prompt, length=300, temperature=0.8, top_k=50)

        return response

    else:
        return "No model found, ending prompting."