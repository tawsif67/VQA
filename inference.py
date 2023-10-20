from config import *
from model import ViltForQuestionAnswering
from label_generation import id2label_dict, label2id_dict

openai.api_key = "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"

def get_completion(prompt, model="gpt-3.5-turbo"):
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=1,  # this is the degree of randomness of the model's output
    )
    return response.choices[0].message["content"]

# Initialize an empty list to store the questions and answers
questions_and_answers = []
#caption = "There are 4 blocks in total, and the colors of the blocks are 1 orange block, 1 gray block, 1 green block, 1 purple block "

def calling_vilt_model(image, question,python device):
    """
    Calling the ViLT model
    """
    model = ViltForQuestionAnswering.from_pretrained(model_name,
                                                    num_labels=len(id2label_dict),
                                                    id2label=id2label_dict,
                                                    label2id=label2id_dict)
    state_dict = torch.load("/content/drive/MyDrive/saved_model.pth")
    # Remove the unexpected key
    if 'vilt.embeddings.text_embeddings.position_ids' in state_dict:
      del state_dict['vilt.embeddings.text_embeddings.position_ids']

# Now you can load the modified state_dict into your model
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    test_encoding = processor(image, question, return_tensors="pt")
    test_encoding = {k: v.to(device) for k,v in test_encoding.items()}
    test_logits = model(**test_encoding).logits
    m = torch.nn.Sigmoid()

    answer = model.config.id2label[test_logits.argmax(-1).item() + 1]

    return answer
user_input = input("User command: ")

# Generate a single prompt to get three questions
prompt = f"""
Peruse the {user_input} and this general description of the environment{caption} Generate three unique questions that you need answer to to better understand the table top environment to process the user
command to generate some pick and place commands.
The questions should be like this: "Where is the yellow block located?" or "Which block is closest/farthest to the green block?"
just generate the question itself. nothing else.
"""

# Use the "get_completion" function to generate a prompt with three questions
prompt_with_three_questions = get_completion(prompt)

# Split the prompt into individual questions
questions = prompt_with_three_questions.strip().split('\n')

# Process each question and obtain the answers
for i, question in enumerate(questions):
    print(f"{question}")
    image_path = "/content/image_1.jpg"
    example_image = Image.open(image_path)
    answer = calling_vilt_model(example_image, question, device)
    questions_and_answers.append(f"Q: {question}\nA: {answer}\n")

# Combine the questions and answers into a single string
qa_string = "\n".join(questions_and_answers)

# Print the generated questions and answers
print(qa_string)

# Get user input for the final step


# Generate a detailed description and commands based on the user input
prompt = f"""
Take the idea of the environment from the following description, {qa_string} and {caption}
and give a detailed description of the environment based on it.
"""
description = get_completion(prompt)
prompt = f"""
Generate one or multiple commands like this: pick the "pick_item"
and place it in the "place location"
based on the {user_input} and description of the environment {description}. don't write anything else other than the commands
"""
final_output = get_completion(prompt)

# Print the final output
print(final_output)