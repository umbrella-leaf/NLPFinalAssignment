import torch
from transformers import BertTokenizer, BertForMaskedLM, RobertaTokenizer, RobertaForMaskedLM
import pandas as pd

# check available device
cuda_able = torch.cuda.is_available()
mps_able = (torch.backends.mps.is_available() and torch.backends.mps.is_built())
device = 'cuda:0' if cuda_able else 'mps' if mps_able else 'cpu'

print(f"Using device: {device}")

# Load BERT model and tokenizer
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForMaskedLM.from_pretrained(model_name)
model.to(device)
model.eval()

model1_name = 'roberta-base'
tokenizer1 = RobertaTokenizer.from_pretrained(model1_name)
model1 = RobertaForMaskedLM.from_pretrained(model1_name)
model1.to(device)
model1.eval()


def log_likelihood(sentence, tokenizer, model, device):
    tokens = tokenizer.encode(sentence, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(tokens, labels=tokens)
    return -outputs.loss.item()


dataset_path = 'dataset/crows_pairs_anonymized.csv'
df = pd.read_csv(dataset_path)

# Sample 3 examples for the case study
gender_df = df[df['bias_type'] == 'gender'].sample(3)

# Filter for age-related sentence pairs
case_study_results = []
case_study_results1 = []
for _, row in gender_df.iterrows():
    stereotype_sentence = row['sent_more']
    astereotype_sentence = row['sent_less']
    ll_stereotype = log_likelihood(stereotype_sentence, tokenizer, model, device)
    ll_astereotype = log_likelihood(astereotype_sentence, tokenizer, model, device)
    case_study_results.append({
        "Sentence (Stereotypical)": stereotype_sentence,
        "Sentence (Astereotypical)": astereotype_sentence,
        "Log-Likelihood Score (Stereo)": ll_stereotype,
        "Log-Likelihood Score (Astereo)": ll_astereotype
    })

    ll1_stereotype = log_likelihood(stereotype_sentence, tokenizer1, model1, device)
    ll1_astereotype = log_likelihood(astereotype_sentence, tokenizer1, model1, device)
    case_study_results1.append({
        "Sentence (Stereotypical)": stereotype_sentence,
        "Sentence (Astereotypical)": astereotype_sentence,
        "Log-Likelihood Score (Stereo)": ll1_stereotype,
        "Log-Likelihood Score (Astereo)": ll1_astereotype
    })

case_study_df = pd.DataFrame(case_study_results)
print(case_study_df.to_markdown(index=False))

case_study_df1 = pd.DataFrame(case_study_results1)
print(case_study_df1.to_markdown(index=False))

# Calculate pseudo-log-likelihood for each pair in the entire dataset
gender_df = df[df['bias_type'] == 'gender']

results = []
results1 = []
for _, row in gender_df.iterrows():
    stereotype_sentence = row['sent_more']
    astereotype_sentence = row['sent_less']
    ll_stereotype = log_likelihood(stereotype_sentence, tokenizer, model, device)
    ll_astereotype = log_likelihood(astereotype_sentence, tokenizer, model, device)
    results.append((ll_stereotype, ll_astereotype))

    ll1_stereotype = log_likelihood(stereotype_sentence, tokenizer1, model1, device)
    ll1_astereotype = log_likelihood(astereotype_sentence, tokenizer1, model1, device)
    results1.append((ll1_stereotype, ll1_astereotype))

# Analyze the results
biased_count = sum(1 for pll_s, pll_a in results if pll_s > pll_a)
total_count = len(results)
bias_percentage = biased_count / total_count * 100

print(f"Bias Percentage: {bias_percentage}% (Ideal: 50%)")

biased_count = sum(1 for pll_s, pll_a in results1 if pll_s > pll_a)
total_count = len(results1)
bias_percentage = biased_count / total_count * 100

print(f"Bias Percentage: {bias_percentage}% (Ideal: 50%)")
