from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch

# check available device
cuda_able = torch.cuda.is_available()
mps_able = (torch.backends.mps.is_available() and torch.backends.mps.is_built())
device = 'cuda:0' if cuda_able else 'mps' if mps_able else 'cpu'

# model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)


def analyze_relationship(sentence_a, sentence_b):
    """
    Function to prompt GPT-2 to analyze the relationship between sentences
    """
    prompt = (f"Determine if the relationship between the following sentences is "
              f"contradictory, entailment, or neutral:\n"
              f"Sentence A: '{sentence_a}'\nSentence B: '{sentence_b}'\n"
              f"Answer:")

    inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)
    outputs = model.generate(inputs, max_length=60, num_return_sequences=1, early_stopping=True)

    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return result.split("Answer:")[1].strip()


# Example sentences
sentence_pairs = [
    ("The cat is sleeping on the mat", "The cat is awake"),
    ("John is a bachelor", "John is married"),
]

for sentence_a, sentence_b in sentence_pairs:
    analysis = analyze_relationship(sentence_a, sentence_b)
    print(f"Sentence A: '{sentence_a}'\nSentence B: '{sentence_b}'\nAnalysis: {analysis}\n")

