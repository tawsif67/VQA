import pandas as pd
import torch
from transformers import ViltConfig
config = ViltConfig.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
from transformers import BertTokenizer, ViltProcessor
from sklearn import metrics
from sklearn.metrics import accuracy_score as accuracy


SEED = 2022
processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-mlm")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Initialize the tokenizer
model_name = "dandelin/vilt-b32-mlm"
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
data_dir = 'data/'
df = pd.read_csv(f'{data_dir}/output.csv')
questions = df['question']
annotations = df['answer']
labels = df['label']
batch_size = 16
num_epochs = 15