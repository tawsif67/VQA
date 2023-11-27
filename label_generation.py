from config import *
import csv

def create_dictionary_from_csv(data_dir):
    result_dict = {}
    label2id_dict= {}
    with open(f"{data_dir}/input.csv", 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            label = int(row['label'])
            answer = row['answer']
            result_dict[label] = answer
            label2id_dict[answer] = label
    return result_dict

def label2id(data_dir):
    result_dict = {}
    label2id_dict= {}
    with open(f"{data_dir}/input.csv", 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            label = int(row['label'])
            answer = row['answer']
            result_dict[label] = answer
            label2id_dict[answer] = label
    return label2id_dict

# Call the function to create the dictionary
dictionary = create_dictionary_from_csv(data_dir)
label2id_dict = label2id(data_dir)
id2label_dict = dictionary
