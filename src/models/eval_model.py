import numpy as np
import pandas as pd
from functools import partial
import nltk
import re
import torch
from datasets import Dataset
import evaluate
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from transformers import pipeline, BartTokenizer, BartForConditionalGeneration, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
from torch.utils.data import DataLoader

from utils.init_settings import set_seed, get_device
from data.preprocess import preprocess
from models.train_model import preprocess_function


seed = set_seed()
device = get_device()

finetuned_directory = "outputs/bart_finetuned_samsum"
model = BartForConditionalGeneration.from_pretrained(finetuned_directory).to(device)
model.eval()

tokenizer = BartTokenizer.from_pretrained(finetuned_directory)
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
rouge = evaluate.load("rouge")

df_test = pd.read_csv("data/raw/samsum-test.csv")
ds_test = preprocess(df_test)

tokenized_test = ds_test.map(partial(preprocess_function, tokenizer=tokenizer), batched=True, remove_columns=ds_test.column_names)

test_loader = DataLoader(
    tokenized_test,
    batch_size=8,
    collate_fn=data_collator
)

predictions = []
references = []

from tqdm import tqdm

for batch in tqdm(test_loader):
    input_ids = batch["input_ids"].to(model.device)
    attention_mask = batch["attention_mask"].to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(input_ids=input_ids, attention_mask=attention_mask)
    
    decoded_preds = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    
    labels = batch["labels"]
    labels[labels == -100] = tokenizer.pad_token_id
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    predictions.extend(decoded_preds)
    references.extend(decoded_labels)

rouge_scores = rouge.compute(predictions=predictions, references=references)
print(rouge_scores)