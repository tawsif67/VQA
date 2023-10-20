from config import *
from label_generation import id2label_dict, label2id_dict

import torch
from transformers import ViltForQuestionAnswering
from torch.utils.data import DataLoader, random_split
from transformers import ViltProcessor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-mlm",
                                                 id2label=config.id2label,
                                                 label2id=config.label2id)
model.to(device)