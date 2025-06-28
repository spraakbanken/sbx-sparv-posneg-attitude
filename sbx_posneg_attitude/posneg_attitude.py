from sparv.api import Annotation, Output, annotator, get_logger
from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch
import os

logger = get_logger(__name__)

@annotator("Classify each words as pos or neg",
           language=["swe"])

def posneg(
    word: Annotation = Annotation("<token:word>"),
    out: Output = Output("<token>:sbx_posneg_attitude.attitude",description="POS or NEG if a token is part of an object of attitude, O if it's not."),
    ):  
    
    #load model
    model_path = 'sbx/KB-bert-base-swedish-cased_posneg_attitude'
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForTokenClassification.from_pretrained(model_path)

    #trim to max len
    splits = trim_to_max_len([val for val in word.read()], tokenizer)

    predictions = [get_predictions(subsplit, tokenizer, model) for subsplit in splits]
    
    # flatten
    predictions = [pred for subsplit in predictions for pred in subsplit]

    # correct invalid BIO-tags 
    predictions = correct_bio_tags(predictions)

    # write predictions out
    out.write([p[1] for p in predictions])


def get_predictions(text,tokenizer,model):

    inputs = tokenizer(
        text,
        is_split_into_words=True,
        return_tensors="pt",
    )
     
    with torch.no_grad():
        outputs = model(**inputs)
    
    logits = outputs.logits

    predictions = torch.argmax(logits, dim=2)

    predicted_token_class = [model.config.id2label[t.item()] for t in predictions[0]]

    #align predicted labels with words

    word_ids = inputs.word_ids(0)

    prev_word_idx = None
    final_preds = []

    for i, word_idx in enumerate(word_ids):
        #if word is none or subword 
        if word_idx is None or word_idx == prev_word_idx:
            continue
        #otherwise word is first in subword
        final_preds.append((text[word_idx], predicted_token_class[i]))

        prev_word_idx = word_idx
    
    return final_preds

def trim_to_max_len(
    text: list,
    tokenizer: AutoTokenizer,
    max_len: int = 512,
    ):
    max_len -= tokenizer.num_special_tokens_to_add()
    subsplits = []
    current_subsplit = []
    current_len = 0
    
    # sum lengths of tokenized tokens until the maximum, split
    for element in text:
        tokenized = tokenizer.tokenize(element)
        current_sublen = len(tokenized)
        if current_sublen == 0:
            continue
        elif (current_len + current_sublen) > max_len:
            subsplits.append(current_subsplit)
            current_subsplit = [element]
            current_len = current_sublen   
        else:  # it's within the length
            current_subsplit.append(element)
            current_len += current_sublen
    subsplits.append(current_subsplit)
       
    return subsplits


def correct_bio_tags(word_tag_list):

    corrected = []
    prev_tag = "O"

    for word, tag in word_tag_list:
        
        if tag == "O":
            corrected.append((word, "O"))
            prev_tag = "O"
            continue

        prefix, tag_type = tag.split("-", 1)

        if prefix == "B":

            # invalid B after B
            if prev_tag == "B":
                corrected.append((word, f"I-{tag_type}"))
                prev_tag = f"I-{tag_type}"
            else:
                corrected.append((word, tag))
                prev_tag = tag

        elif prefix == "I":
            # Check if previous tag is compatible
            if prev_tag == "O" or not prev_tag.endswith(tag_type):
                # invalid I-XXX after O
                corrected.append((word, f"B-{tag_type}"))
                prev_tag = f"B-{tag_type}"
            else:
                corrected.append((word, tag))
                prev_tag = tag

    return corrected
