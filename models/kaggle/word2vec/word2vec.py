# -*- coding:utf-8 -*-

import pandas as pd
import numpy as np
import re
import json
from bs4 import BeautifulSoup

data_path = "../../dataset/data/kaggle/startup/word2vec-nlp-tutorial"
save_file_path = "../../dataset/data/kaggle/startup/word2vec-nlp-tutorial/submission_df.csv"
gnb_word2vec_path = "../../dataset/data/kaggle/startup/word2vec-nlp-tutorial/gnb_word2vec.csv"

train = pd.read_csv('%s/%s' % (data_path, 'labeledTrainData.tsv'), header=0, delimiter="\t", quoting=3)
test = pd.read_csv('%s/%s' % (data_path, 'testData.tsv'), header=0, delimiter="\t", quoting=3)

print(train.shape)
print(train.columns.values)
print(train.head(3))
print(test.head(3))

print('\n 处理前：\n', train['review'][0])
example1 = BeautifulSoup(train['review'][0], "html.parser")

# Use regular expressions to do a find-and-replace
letters_only = re.sub('[^a-zA-Z]',  # 搜寻的pattern
                      ' ',  # 用来替代的pattern(空格)
                      example1.get_text())  # 待搜索的text

print(letters_only)
lower_case = letters_only.lower()  # Convert to lower case
words = lower_case.split()  # Split into word

print('\n处理后: \n', words)


def review_to_wordlist(review):
    reviewText = BeautifulSoup(review, "html.parser").get_text()
    reviewText = re.sub("[^a-zA-Z]", " ", reviewText)
    words = reviewText.lower().split()
    return words


# 预处理数据
label = train['sentiment']
train_data = []
for i in range(len(train['review'])):
    train_data.append(' '.join(review_to_wordlist(train['review'][i])))
test_data = []
for i in range(len(test['review'])):
    test_data.append(' '.join(review_to_wordlist(test['review'][i])))

# 预览数据
print(json.dumps(train_data[0], encoding="UTF-8", ensure_ascii=False), '\n')
print(test_data[0])

from sklearn.feature_extraction.text import TfidfVectorizer as TFIDF

# 参考：http://blog.csdn.net/longxinchen_ml/article/details/50629613

"""
三种方法进行特征处理，会影响到模型结果
单词计数
TF-IDF向量
Word2vec向量
"""

"""
min_df: 最小支持度为2（词汇出现的最小次数）
max_features: 默认为None，可设为int，对所有关键词的term frequency进行降序排序，只取前max_features个作为关键词集
strip_accents: 将使用ascii或unicode编码在预处理步骤去除raw document中的重音符号
analyzer: 设置返回类型
token_pattern: 表示token的正则表达式，需要设置analyzer == 'word'，默认的正则表达式选择2个及以上的字母或数字作为token，标点符号默认当作token分隔符，而不会被当作token
ngram_range: 词组切分的长度范围
use_idf: 启用逆文档频率重新加权
use_idf：默认为True，权值是tf*idf，如果设为False，将不使用idf，就是只使用tf，相当于CountVectorizer了。
smooth_idf: idf平滑参数，默认为True，idf=ln((文档总数+1)/(包含该词的文档数+1))+1，如果设为False，idf=ln(文档总数/包含该词的文档数)+1
sublinear_tf: 默认为False，如果设为True，则替换tf为1 + log(tf)
stop_words: 设置停用词，设为english将使用内置的英语停用词，设为一个list可自定义停用词，设为None不使用停用词，设为None且max_df∈[0.7, 1.0)将自动根据当前的语料库建立停用词表
"""
tfidf = TFIDF(min_df=2,
              max_features=None,
              strip_accents='unicode',
              analyzer='word',
              token_pattern=r'\w{1,}',
              ngram_range=(1, 3),  # 二元文法模型
              use_idf=1,
              smooth_idf=1,
              sublinear_tf=1,
              stop_words='english')  # 去掉英文停用词

# 合并训练和测试集以便进行TFIDF向量化操作
data_all = train_data + test_data
len_train = len(train_data)

tfidf.fit(data_all)
data_all = tfidf.transform(data_all)

# 恢复成训练集和测试集部分
train_x = data_all[:len_train]
test_x = data_all[len_train:]

print('TF-IDF处理结束.')

print("train: \n", np.shape(train_x[0]))
print("test: \n", np.shape(test_x[0]))

from sklearn.naive_bayes import MultinomialNB as MNB

model_NB = MNB()  # (alpha=1.0, class_prior=None, fit_prior=True)
# 为了在预测的时候使用
model_NB.fit(train_x, label)

from sklearn.model_selection import cross_val_score
import numpy as np

print("多项式贝叶斯分类器10折交叉验证得分:  \n", cross_val_score(model_NB, train_x, label, cv=10, scoring='roc_auc'))
print("\n多项式贝叶斯分类器10折交叉验证得分: ", np.mean(cross_val_score(model_NB, train_x, label, cv=10, scoring='roc_auc')))

test_predicted = np.array(model_NB.predict(test_x))
print("saving records...")

submission_df = pd.DataFrame(data={'id': test['id'], 'sentiment': test_predicted})
print(submission_df.head(10))
submission_df.to_csv(save_file_path, columns=['id', 'sentiment'], index=False)

nb_output = pd.DataFrame(data=test_predicted, columns=['sentiment'])
nb_output['id'] = test['id']
nb_output = nb_output[['id', 'sentiment']]
nb_output.to_csv('nb_output.csv', index=False)
print('结束.')

'''
应该是另一种方案
logic regression logistic
'''
from sklearn.linear_model import LogisticRegression as LR
from sklearn.model_selection import GridSearchCV

# 逻辑回归中的grid search参数设置，这个需要研究

grid_values = {'C': [1, 3, 15, 30]}

"""
参数参考这个link
https://www.jianshu.com/p/99ceb640efc5

# penalty: l1 or l2, 用于指定惩罚中使用的标准。
这些parameters都是调参的一部分吧
"""
model_LR = GridSearchCV(LR(penalty='l2', dual=True, random_state=0), grid_values, scoring='roc_auc', cv=20)
model_LR.fit(train_x, label)
# 20折交叉验证
# GridSearchCV(cv=20,
#         estimator=LR(C=1.0,
#             class_weight=None,
#             dual=True,
#             fit_intercept=True,
#             intercept_scaling=1,
#             penalty='l2',
#             random_state=0,
#             tol=0.0001),
#         fit_params={},
#         iid=True,
#         n_jobs=1,
#         param_grid={'C': [30]},
#         pre_dispatch='2*n_jobs',
#         refit=True,
#         scoring='roc_auc',
#         verbose=0)

# 输出结果
# print(model_LR.grid_scores_, '\n', model_LR.best_params_, model_LR.best_params_)
print(model_LR.cv_results_, '\n', model_LR.best_params_, model_LR.best_score_)

'''
{'mean_fit_time': array([0.77368994, 1.95680232, 2.88316183, 3.50976259]), 'std_fit_time': array([0.05099312, 0.19345662, 0.39457327, 0.50422455]), 'mean_score_time': array([0.00273149, 0.0025926 , 0.00262785, 0.00249476]), 'std_score_time': array([0.0001698 , 0.00014623, 0.00014215, 0.00024111]), 'param_C': masked_array(data=[1, 15, 30, 50],
             mask=[False, False, False, False],
       fill_value='?',
            dtype=object), 'params': [{'C': 1}, {'C': 15}, {'C': 30}, {'C': 50}], 'split0_test_score': array([0.95273728, 0.95990784, 0.960192  , 0.9602816 ]), 'split1_test_score': array([0.96081408, 0.96953856, 0.96975104, 0.96994816]), 'split2_test_score': array([0.9583616 , 0.96794112, 0.96825856, 0.96836352]), 'split3_test_score': array([0.95249152, 0.96079104, 0.96123136, 0.96137984]), 'split4_test_score': array([0.96460288, 0.9721088 , 0.9724672 , 0.97263104]), 'split5_test_score': array([0.95881216, 0.96733184, 0.96779008, 0.96797184]), 'split6_test_score': array([0.95679232, 0.96563968, 0.96596736, 0.96606976]), 'split7_test_score': array([0.95171072, 0.96053248, 0.96105216, 0.96125952]), 'split8_test_score': array([0.95526656, 0.9604096 , 0.96051712, 0.96053248]), 'split9_test_score': array([0.94979328, 0.95777024, 0.95817472, 0.95834368]), 'split10_test_score': array([0.95965952, 0.9672192 , 0.9675264 , 0.96764672]), 'split11_test_score': array([0.95329024, 0.96009472, 0.96019712, 0.96021504]), 'split12_test_score': array([0.96268544, 0.97140224, 0.97184256, 0.97202944]), 'split13_test_score': array([0.9571968 , 0.96615936, 0.9666048 , 0.96676864]), 'split14_test_score': array([0.95916544, 0.96551936, 0.96583168, 0.96596992]), 'split15_test_score': array([0.96279296, 0.96956928, 0.96978176, 0.96979968]), 'split16_test_score': array([0.95332096, 0.96132352, 0.96161792, 0.96173568]), 'split17_test_score': array([0.94883328, 0.9570816 , 0.95749632, 0.95771136]), 'split18_test_score': array([0.9528448 , 0.96074496, 0.96114176, 0.9612672 ]), 'split19_test_score': array([0.96429824, 0.97186048, 0.972032  , 0.97212416]), 'mean_test_score': array([0.9567735 , 0.9646473 , 0.9649737 , 0.96510246]), 'std_test_score': array([0.0046911 , 0.00476416, 0.00475249, 0.00475557]), 'rank_test_score': array([4, 3, 2, 1], dtype=int32), 'split0_train_score': array([0.99254593, 1.        , 1.        , 1.        ]), 'split1_train_score': array([0.99230078, 1.        , 1.        , 1.        ]), 'split2_train_score': array([0.9923811, 1.       , 1.       , 1.       ]), 'split3_train_score': array([0.9924227, 1.       , 1.       , 1.       ]), 'split4_train_score': array([0.9923401, 1.       , 1.       , 1.       ]), 'split5_train_score': array([0.9924475, 1.       , 1.       , 1.       ]), 'split6_train_score': array([0.99238184, 1.        , 1.        , 1.        ]), 'split7_train_score': array([0.99249388, 1.        , 1.        , 1.        ]), 'split8_train_score': array([0.99257082, 1.        , 1.        , 1.        ]), 'split9_train_score': array([0.99253744, 1.        , 1.        , 1.        ]), 'split10_train_score': array([0.99235201, 1.        , 1.        , 1.        ]), 'split11_train_score': array([0.99243953, 1.        , 1.        , 1.        ]), 'split12_train_score': array([0.99236668, 1.        , 1.        , 1.        ]), 'split13_train_score': array([0.99248181, 1.        , 1.        , 1.        ]), 'split14_train_score': array([0.99254685, 1.        , 1.        , 1.        ]), 'split15_train_score': array([0.99240575, 1.        , 1.        , 1.        ]), 'split16_train_score': array([0.99240521, 1.        , 1.        , 1.        ]), 'split17_train_score': array([0.99248037, 1.        , 1.        , 1.        ]), 'split18_train_score': array([0.99243375, 1.        , 1.        , 1.        ]), 'split19_train_score': array([0.99242053, 1.        , 1.        , 1.        ]), 'mean_train_score': array([0.99243773, 1.        , 1.        , 1.        ]), 'std_train_score': array([7.34564551e-05, 0.00000000e+00, 2.48253415e-17, 2.48253415e-17])} 
 {'C': 50} 0.965102464

'''

model_LR = LR(penalty='l2', dual=True, random_state=0)
model_LR.fit(train_x, label)

test_predicted = np.array(model_LR.predict(test_x))
print('保存结果...')
submission_df = pd.DataFrame(data={'id': test['id'], 'sentiment': test_predicted})
print(submission_df.head(10))
submission_df.to_csv(save_file_path, columns=['id', 'sentiment'], index=False)
print('结束.')

'''
1. 提交最终的结果到kaggle，AUC为：0.88956，排名260左右，比之前贝叶斯模型有所提高
2. 三元文法，AUC为0.89076
'''

"""
第三种模型
"""
import gensim
import nltk
from nltk.corpus import stopwords
# load google data
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


sentences = []
for i, review in enumerate(train["review"]):
    # print(i, review)
    sentences += review_to_sentences(review, tokenizer, True)

print(np.shape(train["review"]))
print(np.shape(sentences))

unlabeled_train = pd.read_csv("%s/%s" % (data_path, "unlabeledTrainData.tsv"), header=0, delimiter="\t", quoting=3 )
for review in unlabeled_train["review"]:
    sentences += review_to_sentences(review, tokenizer)
print('预处理 unlabeled_train data...')




import time
from gensim.models import Word2Vec

num_features = 300  # Word vector dimensionality
min_word_count = 40  # Minimum word count
num_workers = 4  # Number of threads to run in parallel
context = 10  # Context window size
downsampling = 1e-3  # Downsample setting for frequent words

print("训练模型中...")
model = Word2Vec(sentences, workers=num_workers, \
                 size=num_features, min_count=min_word_count, \
                 window=context, sample=downsampling)
print("训练完成")

print('保存模型...')
model.init_sims(replace=True)
model_name = "%s/%s" % (data_path, "300features_40minwords_10context")
model.save(model_name)
print('保存结束')

# pre-process
model.wv.doesnt_match("man woman child kitchen".split())
model.wv.doesnt_match("france england germany berlin".split())
model.wv.doesnt_match("paris berlin london austria".split())
model.wv.most_similar("man", topn=5)
model.wv.most_similar("queen", topn=5)
model.wv.most_similar("awful", topn=5)
model.wv.most_similar(positive=['woman', 'king'], negative=['man'], topn=1)


def makeFeatureVec(words, model, num_features):
    featureVec = np.zeros((num_features,), dtype='float32')
    nwords = 0.

    # Index2word包含了词表中的所有词，为了检索速度，保存到set中
    index2word_set = set(model.wv.index2word)
    for word in words:
        if word in index2word_set:
            nwords = nwords + 1.
            featureVec = np.add(featureVec, model[word])

    # average
    featureVec = np.divide(featureVec, nwords)
    return featureVec


def getAvgFeatureVecs(reviews, model, num_features):
    count = 0
    reviewFeatureVecs = np.zeros((len(reviews), num_features), dtype="float32")

    for review in reviews:
        if count % 5000 == 0:
            print("Review %d of %d" % (count, len(reviews)))

        reviewFeatureVecs[count] = makeFeatureVec(review, model, num_features)
        count += 1

    return reviewFeatureVecs


trainDataVecs = getAvgFeatureVecs(train_data, model, num_features)
print(np.shape(trainDataVecs))
testDataVecs = getAvgFeatureVecs(test_data, model, num_features)
print(np.shape(testDataVecs))


"""
高斯贝叶斯+Word2vec训练
"""
from sklearn.naive_bayes import GaussianNB as GNB

model_GNB = GNB()
model_GNB.fit(trainDataVecs, label)

#from sklearn.cross_validation import cross_val_score
import numpy as np

print("高斯贝叶斯分类器10折交叉验证得分: ", np.mean(cross_val_score(model_GNB, trainDataVecs, label, cv=10, scoring='roc_auc')))

print('保存结果...')
result = model_GNB.predict( testDataVecs )
submission_df = pd.DataFrame(data ={'id': test['id'], 'sentiment': result})
print(submission_df.head(10))
submission_df.to_csv(gnb_word2vec_path,columns = ['id','sentiment'], index = False)
print('结束.')

"""
从验证结果来看，没有超过基于TF-IDF多项式贝叶斯模型
"""

# 加载训练好的词向量

from gensim.models.word2vec import Word2Vec
model = Word2Vec.load_word2vec_format("vector.txt", binary=False)  # C text format
# model = Word2Vec.load_word2vec_format("vector.bin", binary=True)  # C
from gensim.models.word2vec import Word2Vec
model = Word2Vec.load_word2vec_format("GoogleNews-vectors-negative300.bin", binary=True)


# 测试预测效果

print(model.most_similar(positive=["woman", "king"], negative=["man"], topn=5))
print(model.most_similar(positive=["biggest", "small"], negative=["big"], topn=5))
print(model.most_similar(positive=["ate", "speak"], negative=["eat"], topn=5))










