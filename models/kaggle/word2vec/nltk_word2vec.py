# -*- coding:utf-8 -*-

import pandas as pd
import numpy as np
import re
import json
from bs4 import BeautifulSoup
'''
下面是第三种特种构建方法
神经网络语言模型L = SUM[log(p(w|contect(w))]，即在w的上下文下计算当前词w的概率，
由公式可以看到，我们的核心是计算p(w|contect(w)， Word2vec给出了构造这个概率的一个方法。
'''
import gensim
import nltk
from nltk.corpus import stopwords

tokenizer = nltk.data.load('/opt/data/nlp/nltk_data/tokenizers/punkt/english.pickle')


def review_to_wordlist(review, remove_stopwords=False):
    # review = BeautifulSoup(review, "html.parser").get_text()
    review_text = re.sub("[^a-zA-Z]", " ", review)

    words = review_text.lower().split()

    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]
    # print(words)
    return (words)


def review_to_sentences(review, tokenizer, remove_stopwords=False):
    '''
    1. 将评论文章，按照句子段落来切分(所以会比文章的数量多很多)
    2. 返回句子列表，每个句子由一堆词组成
    '''
    review = BeautifulSoup(review, "html.parser").get_text()
    # raw_sentences 句子段落集合
    raw_sentences = tokenizer.tokenize(review)
    # print(raw_sentences)

    sentences = []
    for raw_sentence in raw_sentences:
        if len(raw_sentence) > 0:
            # 获取句子中的词列表
            sentences.append(review_to_wordlist(raw_sentence, remove_stopwords))
    return sentences


