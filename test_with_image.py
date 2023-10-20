from inference import  calling_vilt_model
from PIL import Image
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
questions = "Where is the gray block?"
example_image = Image.open("data\\vqa_data\\vqa_data\image_13.png")
answer = calling_vilt_model(example_image, questions, device)