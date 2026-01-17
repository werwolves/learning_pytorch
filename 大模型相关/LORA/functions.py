

def get_loader(text_lens=100):
    import torch
    import random
    from transformers import BertTokenizer  
    from datasets import Dataset
    
    tokenizer = BertTokenizer(vocab_file=r"大模型相关\LORA\tokenizer\vocab.txt",
                              model_max_length=512,
                              )
    
    
    
    
    
    
    
    
    
    print("Preparing data loader...")
    

get_loader()

























