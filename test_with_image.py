questions = "Which thing is farthest from red bowl?"
example_image = Image.open("/content/images/image_2.jpg")
answer = calling_vilt_model(example_image, questions, device)