
from config import *
from label_generation import id2label_dict
import torch
from PIL import Image
import pandas as pd
from transformers import BertTokenizer

class VQADataset(torch.utils.data.Dataset):
    """VQA (v2) dataset."""
    def __init__(self, df, processor, tokenizer):
    # def __init__(self, questions, annotations, processor):
        self.df = df
        self.questions = df['question']
        self.annotations = df['answer']
        self.labels = df['label']
        self.processor = processor
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        # get image + text
        annotation = self.annotations[idx]
        questions = self.questions[idx]
        img_id = self.df['image'][idx]
        image = Image.open(f"{data_dir}/vqa_data/{img_id}")
        labels = self.labels[idx]
        text = questions
        targets = torch.zeros(len(id2label_dict))
        targets[labels-1] = 1
        encoding = self.processor(image, text, padding="max_length", truncation=True, return_tensors="pt")
        # remove batch dimension
        for k,v in encoding.items():
          # print(k)
          encoding[k] = v.squeeze()
        #targets = torch.zeros(len(id2label_dict))
        tokens = tokenizer.tokenize(annotation)
        #tokens = [int(token) for token in tokens]
        #print(tokens)

        encoding["labels"] = targets
        return encoding