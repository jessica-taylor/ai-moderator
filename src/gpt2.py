import torch
import math
import time
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import functools
import re
import sys
import numpy as np

class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout
        self.log = open("log.dat", "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)  

    def flush(self):
        self.log.flush()

sys.stdout = Logger()


def to_segments(transcript):
    # transcript = transcript.replace('\n', '`')
    # return re.findall(r'(.*)\s+(\d+:\d+)', transcript, re.MULTILINE)
    # return re.findall(r'([\w ]+)\s+(\d+:\d+)\s+`([^`]+)', transcript, re.MULTILINE)
    segs = re.findall(r'([\w ]+)\s+(\d+:\d+)\s+\n([^\n]+)', transcript, re.MULTILINE)
    # return [(s[0], s[2]) for s in segs]
    return segs




tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

def score(string):
    tensor = tokenizer.encode(string, return_tensors="pt")
    loss=model(tensor, labels=tensor)[0]
    return loss.cpu().detach().numpy()

def rel_score(prompt, text):
    return score(prompt + ' ' + text) - score(text)

# Returns tokenized version of string for debugging
def tokenize(string):
    print([tokenizer.decode(i) for i in tokenizer.encode(string)])
    return [tokenizer.decode(i) for i in tokenizer.encode(string)]


# Returns raw predictions for next token relative to string
def get_logits(string):
    inputs = tokenizer(string, return_tensors="pt")
    outputs = model(**inputs, labels=inputs["input_ids"])

    logits = outputs.logits

    return logits


# Returns probabilities for next token relative to string
def get_probs(string):
    inputs = tokenizer(string, return_tensors="pt")
    outputs = model(**inputs, labels=inputs["input_ids"])

    probs = torch.nn.Softmax(dim=2)(outputs.logits)

    return probs


# Returns the most likely next token relative to string
def most_prob_cont(string):
    probs = get_probs(string)

    return tokenizer.decode(torch.argmax(probs[0][-1])), torch.log(torch.max(probs[0][-1]))


# Returns the greedy sequence continuation of length seq_len relative to string
def greedy_seq(string, seq_len):
    model = GPT2LMHeadModel.from_pretrained("gpt2", pad_token_id=tokenizer.eos_token_id)

    inputs = tokenizer.encode(string, return_tensors="pt")
    outputs = model.generate(inputs, max_length=seq_len+len(inputs[0]))

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


# Returns the entropy value (mean of logit values of comprising token(s)) for a sequence of next tokens relative to string, and number of tokens
@functools.cache
def entropy_of_next(string, next_str, str_lim=1000):
    # if string == "":
    #     tokens = tokenizer.encode(next_str)
    #     if len(tokens) == 1:
    #         raise ValueError("Prefix string argument to function should not be empty; or barring that, next_str argument should be of multiple tokens")
    #     else:
    #         first_token = tokenizer.decode(tokens[0])
    #         string = first_token
    #         next_str = next_str[len(first_token):]
    # elif len(string) > str_lim:
    #     # Cutting string to get latest set of sentences of total length < str_lim
    #     string = string[-(str_lim - string[-str_lim:].find(".")) + 2:]

    if string == '':
        string = 'text:'

    print('ENTROPY OF NEXT', string, '|||', next_str)



    next_tokens = tokenizer.encode(next_str)

    running_sum = []
    for token in next_tokens:
        if len(string) > 200:
            string = string[string[-200:].find("."):]
        probs = get_probs(string)
        print('PROBS', string, token)
        running_sum.append(torch.log(probs[0][-1][token]))
        string += tokenizer.decode(token)

    return (-sum(running_sum).item(), len(running_sum))


# Returns a vector of entropy values (logit values) for all comprising token(s) of a sequence of next tokens relative to string
def entropy_of_next_full(string, next_str):
    probs = get_probs(string)

    next_tokens = tokenizer.encode(next_str)

    running_sum = []
    for token in next_tokens:
        running_sum.append(torch.log(probs[0][-1][token]))

        string += tokenizer.decode(token)
        probs = get_probs(string)

    return running_sum


# Returns the probability of next_word being continuation to string
def word_prob(string, next_word):
    probs = get_probs(string)

    next_token = tokenizer.encode(next_word)

    return torch.mean(probs[0][-1][next_token]).item()

# a segment is represented as (speaker, utterance), both strings.

def mutual_info(str1, str2):
    # return entropy_of_next('text:', str2)[0] - entropy_of_next('text: ' + str1, str2)[0]
    return rel_score('text:', str2) - rel_score('text: ' + str1, str2)

def mutual_infos(segments, back=2):
    infos = []
    for i in range(len(segments)):
        infos_i = []
        for j in range(max(0, i-back), i):
            infos_i.append(mutual_info(segments[j][1], segments[i][1]))
        infos.append(infos_i)
    return infos

if __name__ == '__main__':
    transcript = open('transcripts/joe_rogan_1258_15m.txt').read()
    segments = to_segments(transcript)
    back = 2
    for i in range(len(segments)):
        print('#####################################')
        speaker, time, text = segments[i]
        print(i, speaker, time, text)
        for j in range(max(0, i-back), i):
            mut = mutual_info(segments[j][2], text)
            print(j, mut)

    # segments = [
    #         ('Bob', "What's up with electric cars these days?"),
    #         ('Joe', "Elon's working on some new stuff with better batteries."),
    #         ('Fred', "It's just the weather changing."),
    #         ('Carol', "Ford and GM are also making more efficient power systems for vehicles.")
    #         ]
    # print(zip(segments, mutual_infos(segments)))
