from config import *
import csv

def create_dictionary_from_csv(csv_file_path):
    result_dict = {}
    label2id_dict= {}
    with open('/content/output.csv', 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            label = int(row['label'])
            answer = row['answer']
            result_dict[label] = answer
            label2id_dict[answer] = label
    return result_dict

def label2id(csv_file_path):
    result_dict = {}
    label2id_dict= {}
    with open('/content/output.csv', 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            label = int(row['label'])
            answer = row['answer']
            result_dict[label] = answer
            label2id_dict[answer] = label
    return label2id_dict

# Provide the path to your CSV file


# Call the function to create the dictionary
dictionary = create_dictionary_from_csv(csv_file_path)
label2id_dict = label2id(csv_file_path)
id2label_dict = dictionary
# Print the resulting dictionary
print(dictionary)