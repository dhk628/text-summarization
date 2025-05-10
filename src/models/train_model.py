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

from utils.init_settings import set_seed, get_device
from data.preprocess import preprocess


def compute_metrics(eval_pred, tokenizer, rouge):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
    decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]

    result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)

    return {k: round(v, 4) for k, v in result.items()}


def preprocess_function(examples, tokenizer):
    inputs = [doc for doc in examples["dialogue"]]
    model_inputs = tokenizer(inputs, max_length=1024, truncation=True)

    labels = tokenizer(text_target=examples["summary"], max_length=128, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


if __name__ == "__main__":
    seed = set_seed()
    device = get_device()

    df_train = pd.read_csv("data/raw/samsum-train.csv")
    df_test = pd.read_csv("data/raw/samsum-test.csv")
    df_val = pd.read_csv("data/raw/samsum-validation.csv")

    ds_train = preprocess(df_train)
    ds_test = preprocess(df_test)
    ds_val = preprocess(df_val)

    checkpoint = "facebook/bart-large-xsum"

    tokenizer = BartTokenizer.from_pretrained(checkpoint)
    model = BartForConditionalGeneration.from_pretrained(checkpoint).to(device)
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
    rouge = evaluate.load("rouge")

    tokenized_train = ds_train.map(partial(preprocess_function, tokenizer=tokenizer), batched=True, remove_columns=ds_train.column_names)
    tokenized_val = ds_val.map(partial(preprocess_function, tokenizer=tokenizer), batched=True, remove_columns=ds_val.column_names)
    tokenized_test = ds_test.map(partial(preprocess_function, tokenizer=tokenizer), batched=True, remove_columns=ds_test.column_names)

    training_args = Seq2SeqTrainingArguments(
        output_dir='outputs/bart_samsum_checkpoint',
        eval_strategy='epoch',
        save_strategy='epoch',
        load_best_model_at_end=True,
        metric_for_best_model='eval_loss',
        seed=seed,
        learning_rate=2e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=2,
        weight_decay=0.01,
        save_total_limit=2,
        num_train_epochs=4,
        predict_with_generate=True,
        fp16=True,
        report_to="none"
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=partial(compute_metrics, tokenizer=tokenizer, rouge=rouge),
    )

    trainer.train()

    directory = "outputs/bart_finetuned_samsum_2"
    trainer.save_model(directory)
    tokenizer.save_pretrained(directory)