from config import *
from label_generation import id2label_dict
import torch
from PIL import Image
import pandas as pd
from transformers import BertTokenizer

class VQADataset(torch.utils.data.Dataset):
    """VQA (v2) dataset."""
    def __init__(self, df, processor, tokenizer):
        self.df = df
        self.questions = df['question']
        self.annotations = df['answer']
        self.labels = df['label']
        self.num_labels = len(id2label_dict)  # Number of labels
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

        # One-hot encode the labels
        #targets = torch.zeros(self.num_labels)
        targets = torch.zeros(len(id2label_dict))
        targets[labels-1] = 1 

        encoding = self.processor(image, text, padding="max_length", truncation=True, return_tensors="pt")
        # remove batch dimension
        for k,v in encoding.items():
            encoding[k] = v.squeeze()

        tokens = self.tokenizer.tokenize(annotation)
        
        encoding["labels"] = targets
        # print(labels)
        return encoding