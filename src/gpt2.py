import torch
import math
import time
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import functools
import re
import sys
import numpy as np
from datasets import load_dataset
from tqdm import tqdm

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



tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

transcript = open('transcripts/joe_rogan_1258_15m.txt').read()

@functools.cache
def surprisal(string):
    # https://huggingface.co/docs/transformers/perplexity
    encodings = tokenizer(string, return_tensors='pt')
    max_length = model.config.n_positions
    stride = 512
    seq_len = encodings.input_ids.size(1)
    nlls = []
    prev_end_loc = 0
    # for begin_loc in tqdm(range(0, seq_len, stride)):
    for begin_loc in range(0, seq_len, stride):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
        input_ids = encodings.input_ids[:, begin_loc:end_loc]
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)

            # loss is calculated using CrossEntropyLoss which averages over input tokens.
            # Multiply it with trg_len to get the summation instead of average.
            # We will take average over all the tokens to get the true average
            # in the last step of this example.
            neg_log_likelihood = outputs.loss * trg_len

        nlls.append(neg_log_likelihood)

        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

    return torch.stack(nlls).sum().item()


def to_segments(transcript):
    # transcript = transcript.replace('\n', '`')
    # return re.findall(r'(.*)\s+(\d+:\d+)', transcript, re.MULTILINE)
    # return re.findall(r'([\w ]+)\s+(\d+:\d+)\s+`([^`]+)', transcript, re.MULTILINE)
    segs = re.findall(r'([\w ]+)\s+(\d+:\d+)\s+\n([^\n]+)', transcript, re.MULTILINE)
    # return [(s[0], s[2]) for s in segs]
    return segs



# Returns the entropy value (mean of logit values of comprising token(s)) for a sequence of next tokens relative to string, and number of tokens
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

    # print('ENTROPY OF NEXT', string, '|||', next_str)

    return surprisal(string + ' ' + next_str) - surprisal(string)


    # next_tokens = tokenizer.encode(next_str)

    # running_sum = []
    # for token in next_tokens:
    #     if len(string) > 200:
    #         string = string[string[-200:].find("."):]
    #     probs = get_probs(string)
    #     # print('PROBS', string, token)
    #     running_sum.append(torch.log(probs[0][-1][token]))
    #     string += tokenizer.decode(token)

    # return (-sum(running_sum).item(), len(running_sum))


# a segment is represented as (speaker, utterance), both strings.

def mutual_info(str1, str2):
    return surprisal(str1) + surprisal(str2) - surprisal(str1 + ' ' + str2)

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
    # segments = [
    #         ('Bob', '...', "What's up with electric cars these days?"),
    #         ('Joe', '...', "Elon's working on some new stuff with better batteries."),
    #         ('Fred', '...', "It's just the weather changing."),
    #         ('Carol', '...', "Ford and GM are also making more efficient power systems for vehicles.")
    #         ]
    back = 2
    for i in range(len(segments)):
        print('#####################################')
        speaker, time, text = segments[i]
        print(i, speaker, time, text)
        for j in range(max(0, i-back), i):
            mut = mutual_info(segments[j][2], text)
            print(j, mut)

    # print(zip(segments, mutual_infos(segments)))
