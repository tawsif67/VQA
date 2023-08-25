from config import *
from label_generation import id2label_dict, label2id_dict

import torch
from transformers import ViltForQuestionAnswering
from torch.utils.data import DataLoader, random_split
from transformers import ViltProcessor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ViltForQuestionAnswering.from_pretrained(model_name,
                                                 num_labels=len(id2label_dict),
                                                 id2label=id2label_dict,
                                                 label2id=label2id_dict)
model.to(device)