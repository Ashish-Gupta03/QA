# from __future__ import unicode_literals
import argparse
import os
import sys
import json
import time
import re
import pickle
import spacy
import numpy as np
from unidecode import unidecode
from collections import Counter
from tqdm import tqdm
import random
import gc

from gensim.models import KeyedVectors
from keras.preprocessing.sequence import pad_sequences
# from scripts.retriever import interactive
# import tensorflow as tf
# from drqa import retriever
from extract_text_given_time import extract_data


nlp = spacy.blank("en")

def word_tokenize(sent):
    doc = nlp(sent)
    return [token.text for token in doc]

def convert_idx(text, tokens):
    current = 0
    spans = []
    for token in tokens:
        current = text.find(token, current)
        if current < 0:
            print("Token {} cannot be found".format(token))
            raise Exception()
        spans.append((current, current + len(token)))
        current += len(token)
    return spans

# ------------------------------------------------------------------------------
# Process dataset examples
# ------------------------------------------------------------------------------


def word2vec(word2vec_path):
    # Download word2vec data if it's not present yet
    # if not path.exists(word2vec_path):
    #     glove_file_path = os.getcwd()+'/glove.840B.300d.txt'
    #     print('Converting Glove to word2vec...', end='')
    #     glove2word2vec(glove_file_path, word2vec_path)  # Convert glove to word2vec
    #     os.remove(glove_file_path)                      # Remove glove file and keep only word2vec
    #     print('Done')

    print('Reading word2vec data... ', end='')
    model = KeyedVectors.load_word2vec_format(word2vec_path)
    print('Done')

    def get_word_vector(word):
        try:
            return model[word]
        except KeyError:
            return np.zeros(model.vector_size)

    return get_word_vector

# def process_file(filename, data_type, word_counter, char_counter):
#     print("Generating {} examples...".format(data_type))
#     examples = []
#     eval_examples = {}
#     total = 0
#     q_id = 1
#     print('type is ',type(filename))
#     with open(filename, "r") as fh:
#         source = json.load(fh)['data']
#         for article in tqdm(source):
#             q_id += 1
#         # for article in tqdm(source["data"]):
#             # for para in article["paragraphs"]:
#             if article["videoId"]!="":
#                 os.chdir('/media/ashish/New Volume/M.Tech/Thesis/Dataset/bosch_srt_orig/all_merged_output_new')
#                 context = open(os.path.join(os.getcwd(),article["videoId"])).read().replace(
#                     "''", '" ').replace("``", '" ')
#             else:
#                 os.chdir('/media/ashish/New Volume/M.Tech/Thesis/Dataset/Thesis2/scripts/retriever/output')
#                 ranker = retriever.get_class('tfidf')(tfidf_path='videoken_new-tfidf-ngram=2-hash=16777216-tokenizer=simple.npz')
#                 doc_names, doc_scores = ranker.closest_docs(article['question'], 1)
#                 os.chdir('/media/ashish/New Volume/M.Tech/Thesis/Dataset/bosch_srt_orig/all_merged_output_new')
#                 # print('doc_names ',doc_names)
#                 # print('question ',article['question'])
#                 context = open(os.path.join(os.getcwd(),doc_names[0])).read().replace("''", '" ').replace("``", '" ')
#             context_tokens = word_tokenize(context)
#             context_chars = [list(token) for token in context_tokens]            
#             spans = convert_idx(context, context_tokens)
#             # print('spans type ',type(spans))
#             # print('span is ',spans[0])
#             for token in context_tokens:
#                 word_counter[token] += 1#len(para["qas"])
#                 for char in token:
#                     char_counter[char] += 1#len(para["qas"])
#             # for qa in para["qas"]:
#             total += 1
#             ques = article["question"].replace("''", '" ').replace("``", '" ')
#             ques_tokens = word_tokenize(ques)
#             ques_chars = [list(token) for token in ques_tokens]
#             for token in ques_tokens:
#                 word_counter[token] += 1
#                 for char in token:
#                     char_counter[char] += 1
#             y1s, y2s = [], []
#             answer_texts = []
#             # for answer in qa["answers"]:
#             answer_text = article['answer']
#             # answer_start = len(article['Start'])
#             # answer_end = len(article['End'])
#             answer_texts.append(answer_text)
#             # answer_span = []
#             # for idx, span in enumerate(spans):
#             #     if not (answer_end <= span[0] or answer_start >= span[1]):
#             #         print('idx ',idx)
#             #         answer_span.append(idx)
#             # y1, y2 = answer_span[0], answer_span[-1]
#             # y1s.append(y1)
#             # y2s.append(y2)
#             # print('y1s ',y1s)
#             # print('y2s ',y2s)
#             example = {"context_tokens": context_tokens, "context_chars": context_chars, "ques_tokens": ques_tokens,
#                        "ques_chars": ques_chars, "y1s": article['Start'], "y2s": article['End'], "id": total}
#             examples.append(example)
#             eval_examples[str(total)] = {
#                 "context": context, "spans": spans, "answers": answer_texts, "uuid": q_id}
#     random.shuffle(examples)
#     print("{} questions in total".format(len(examples)))
#     return examples, eval_examples


# def get_embedding(counter, data_type, limit=-1, emb_file=None, size=None, vec_size=None):
#     print("Generating {} embedding...".format(data_type))
#     embedding_dict = {}
#     filtered_elements = [k for k, v in counter.items() if v > limit]
#     if emb_file is not None:
#         assert size is not None
#         assert vec_size is not None
#         with open(emb_file, "r", encoding="utf-8") as fh:
#             for line in tqdm(fh, total=size):
#                 array = line.split()
#                 # print('array[0:-vec_size] is ',array[0:-vec_size])
#                 word = "".join(array[0:-vec_size])
#                 vector = list(map(float, array[-vec_size:]))
#                 if word in counter and counter[word] > limit:
#                     embedding_dict[word] = vector
#         print("{} / {} tokens have corresponding {} embedding vector".format(
#             len(embedding_dict), len(filtered_elements), data_type))
#     else:
#         assert vec_size is not None
#         for token in filtered_elements:
#             embedding_dict[token] = [np.random.normal(
#                 scale=0.1) for _ in range(vec_size)]
#         print("{} tokens have corresponding embedding vector".format(
#             len(filtered_elements)))

#     NULL = "--NULL--"
#     OOV = "--OOV--"
#     token2idx_dict = {token: idx for idx,
#                       token in enumerate(embedding_dict.keys(), 2)}
#     token2idx_dict[NULL] = 0
#     token2idx_dict[OOV] = 1
#     embedding_dict[NULL] = [0. for _ in range(vec_size)]
#     embedding_dict[OOV] = [0. for _ in range(vec_size)]
#     idx2emb_dict = {idx: embedding_dict[token]
#                     for token, idx in token2idx_dict.items()}
#     emb_mat = [idx2emb_dict[idx] for idx in range(len(idx2emb_dict))]
#     return emb_mat, token2idx_dict



# def build_features(examples, data_type, out_file, word2idx_dict, char2idx_dict, is_test=False):

#     para_limit = 1000 if is_test else 400
#     ques_limit = 100 if is_test else 50
#     char_limit = 16

#     def filter_func(example, is_test=False):
#         return len(example["context_tokens"]) > para_limit or len(example["ques_tokens"]) > ques_limit

#     print("Processing {} examples...".format(data_type))
#     writer = tf.python_io.TFRecordWriter(out_file)
#     total = 0
#     total_ = 0
#     meta = {}
#     for example in tqdm(examples):
#         total_ += 1

#         if filter_func(example, is_test):
#             continue

#         total += 1
#         context_idxs = np.zeros([para_limit], dtype=np.int32)
#         context_char_idxs = np.zeros([para_limit, char_limit], dtype=np.int32)
#         ques_idxs = np.zeros([ques_limit], dtype=np.int32)
#         ques_char_idxs = np.zeros([ques_limit, char_limit], dtype=np.int32)
#         y1 = np.ones([para_limit], dtype=np.float32)
#         y2 = np.ones([para_limit], dtype=np.float32)

#         def _get_word(word):
#             for each in (word, word.lower(), word.capitalize(), word.upper()):
#                 if each in word2idx_dict:
#                     return word2idx_dict[each]
#             return 1

#         def _get_char(char):
#             if char in char2idx_dict:
#                 return char2idx_dict[char]
#             return 1

#         for i, token in enumerate(example["context_tokens"]):
#             context_idxs[i] = _get_word(token)

#         for i, token in enumerate(example["ques_tokens"]):
#             ques_idxs[i] = _get_word(token)

#         for i, token in enumerate(example["context_chars"]):
#             for j, char in enumerate(token):
#                 if j == char_limit:
#                     break
#                 context_char_idxs[i, j] = _get_char(char)

#         for i, token in enumerate(example["ques_chars"]):
#             for j, char in enumerate(token):
#                 if j == char_limit:
#                     break
#                 ques_char_idxs[i, j] = _get_char(char)

#         # print('examples ',example)
#         # start, end = example["y1s"][-1], example["y2s"][-1]
#         # y1[0], y2[0] = 1.0, 1.0

#         record = tf.train.Example(features=tf.train.Features(feature={
#                                   "context_idxs": tf.train.Feature(bytes_list=tf.train.BytesList(value=[context_idxs.tostring()])),
#                                   "ques_idxs": tf.train.Feature(bytes_list=tf.train.BytesList(value=[ques_idxs.tostring()])),
#                                   "context_char_idxs": tf.train.Feature(bytes_list=tf.train.BytesList(value=[context_char_idxs.tostring()])),
#                                   "ques_char_idxs": tf.train.Feature(bytes_list=tf.train.BytesList(value=[ques_char_idxs.tostring()])),
#                                   "y1": tf.train.Feature(bytes_list=tf.train.BytesList(value=[y1.tostring()])),
#                                   "y2": tf.train.Feature(bytes_list=tf.train.BytesList(value=[y2.tostring()])),
#                                   "id": tf.train.Feature(int64_list=tf.train.Int64List(value=[example["id"]]))
#                                   }))
#         writer.write(record.SerializeToString())
#     print("Build {} / {} instances of features in total".format(total, total_))
#     meta["total"] = total
#     writer.close()
#     return meta

# -----------------------------------------------------------------------------
# Commandline options
# -----------------------------------------------------------------------------


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--word2vec_path', type=str,
                            default='data/word2vec_from_glove_300.vec',
                            help='Word2Vec vectors file path')
    parser.add_argument('--outfile', type=str, default='data/tmp.pkl',
                            help='Desired path to output pickle')
    parser.add_argument('--include_str', action='store_true',
                        help='Include strings')
    parser.add_argument('data', type=str, help='Data json')
    args = parser.parse_args()

    if not args.outfile.endswith('.pkl'):
            args.outfile += '.pkl'

    cwd = os.getcwd()
    print('Reading NPTEL data... ', end='')
    with open(args.data) as fd:
        samples = json.load(fd)['data']
    print('Done!')

    word_vector = word2vec(args.word2vec_path)
    def parse_sample(sample):
        inputs = []
        targets = []

        vId = ''
        if sample["videoId"]!="":
            os.chdir('/media/ashish/New Volume/M.Tech/Thesis/Dataset/bosch_srt_orig/all_merged_output_new')
            print('current wd ',os.getcwd())
            context = open(os.path.join(os.getcwd(),sample["videoId"])).read().replace(
                    "''", '" ').replace("``", '" ')        
            vId = sample['videoId']
        else:
                os.chdir('/media/ashish/New Volume/M.Tech/Thesis/Dataset/Thesis2/scripts/retriever/output')
                ranker = retriever.get_class('tfidf')(tfidf_path='videoken_new-tfidf-ngram=2-hash=16777216-tokenizer=simple.npz')
                doc_names, doc_scores = ranker.closest_docs(sample['question'], 1)
                os.chdir('/media/ashish/New Volume/M.Tech/Thesis/Dataset/bosch_srt_orig/all_merged_output_new')                
                context = open(os.path.join(os.getcwd(),doc_names[0])).read().replace("''", '" ').replace("``", '" ')
                vId = doc_names[0]

        tokens = word_tokenize(context)
        # context_chars = [list(token) for token in tokens]
        char_offsets = convert_idx(context, tokens)
	    # print('sample videoid ',sample["videoId"])
        # print('tokens ',tokens)
        # print('spans ',char_offsets)

        text,answer_start = extract_data(vId,sample['Start'],sample['End'])
        answer_end = len(text)+ answer_start
        
        try:
            answer_start = [s <= answer_start < e
                            for s, e in char_offsets].index(True)
            targets.append(answer_start)
            answer_end   = [s <= answer_end < e
                            for s, e in char_offsets].index(True)
            targets.append(answer_end)
        except ValueError:
            return None


        # print('targets ',targets)
        tokens = [unidecode(token) for token in tokens]

        context_vecs = [word_vector(token) for token in tokens]
        context_vecs = np.vstack(context_vecs).astype(np.float32)
        inputs.append(context_vecs)

        if args.include_str:
            context_str = [np.fromstring(token, dtype=np.uint8).astype(np.int32)
                           for token in tokens]
            context_str = pad_sequences(context_str, maxlen=25)
            inputs.append(context_str)

        # print('question ',question)
        question = sample["question"].replace("''", '" ').replace("``", '" ')
        tokens = word_tokenize(question)
        # context_chars = [list(token) for token in tokens]
        char_offsets = convert_idx(question, tokens)
        # char_offsets = [list(token) for token in tokens]
        tokens = [unidecode(token) for token in tokens]

        question_vecs = [word_vector(token) for token in tokens]
        question_vecs = np.vstack(question_vecs).astype(np.float32)
        inputs.append(question_vecs)

        if args.include_str:
            question_str = [np.fromstring(token, dtype=np.uint8).astype(np.int32)
                            for token in tokens]
            question_str = pad_sequences(question_str, maxlen=25)
            inputs.append(question_str)

        return [inputs, targets]

    print('Parsing samples... ', end='')

    samples = [parse_sample(sample) for sample in tqdm(samples)]
    samples = [sample for sample in samples if sample is not None]
    print('Done!')

    gc.collect()
    # Transpose
    def transpose(x):
        return map(list, zip(*x))

    data = [transpose(input) for input in transpose(samples)]

    os.chdir(cwd)
    print('Writing to file {}... '.format(args.outfile), end='')
    with open(args.outfile, 'wb') as fd:
        pickle.dump(data, fd, protocol=pickle.HIGHEST_PROTOCOL)
    print('Done!')
