from config import *
from model import ViltForQuestionAnswering
from label_generation import id2label_dict, label2id_dict

def calling_vilt_model(image, question, device):
    """
    Calling the ViLT model
    """
    model = ViltForQuestionAnswering.from_pretrained(model_name,
                                                     num_labels=len(id2label_dict),
                                                     id2label=id2label_dict,
                                                     label2id=label2id_dict)
    model.load_state_dict(torch.load("saved_model.pth", map_location=device))
    model.to(device)
    model.eval()

    test_encoding = processor(image, question, return_tensors="pt")
    test_encoding = {k: v.to(device) for k,v in test_encoding.items()}
    test_logits = model(**test_encoding).logits
    m = torch.nn.Sigmoid()
    
    print(torch.max(m(test_logits)))
    print((m(test_logits)))
    
    answer = model.config.id2label[test_logits.argmax(-1).item()+1]
    print(test_logits.argmax(-1))
    print(torch.max(m(test_logits)))
    print((m(test_logits)))
    print("\n\033[1;31;34m>> Predicted answer =", answer)

    return answer