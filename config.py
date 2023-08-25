import pandas as pd
from transformers import ViltConfig
config = ViltConfig.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
from transformers import BertTokenizer
from sklearn import metrics
from sklearn.metrics import accuracy


SEED = 2022
processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-mlm")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Initialize the tokenizer
model_name = "dasndelin/vilt-b32-mlm"
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
csv_file_path = '/content'
df = pd.read_csv(f'{csv_file_path}/output.csv')
questions = df['question']
#print(questions)
annotations = df['answer']
#print(annotations)
labels = df['label']
batch_size = 16
num_epochs = 15