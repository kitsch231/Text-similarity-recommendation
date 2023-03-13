'''这个代码用来训练tfidf模型来比较相似度'''

import pandas as pd
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import pickle
import emoji
import re
import pandas as pd
from tqdm import tqdm
import jieba


with open('../data/stop_words.txt','r',encoding='utf-8')as f:
    stopwords=f.readlines()
    stopwords=[x.strip() for x in stopwords]
#print(stopwords)

def get_words(sen):
    #print(sen)
    new_words=[]
    sen=sen.replace(' ','').replace('/n','').replace('/t','').replace('\t','').replace('\n','')
    sen = emoji.demojize(sen)#去除emoji表情
    sen = re.sub(':\S+?:', '', sen)
    sen=sen.replace(' ','')

    results = re.compile(r'[http|https]*://[a-zA-Z0-9.?/&=:]*', re.S)
    sen = re.sub(results, '', sen)

    #cutwords1 = jieba.lcut(sen)  # 分词
    cutwords1 = jieba.lcut(sen)  # 分词

    #print(cutwords1)
    interpunctuations = [',','.','’','…','-',':','~', ';', '"','?',"'s", '(', ')','...', '[', ']', '&', '!', '*', '@', '#', '$', '%']  # 定义符号列表
    cutwords2 = [word for word in cutwords1 if word not in interpunctuations]  # 去除标点符号
    stops = set(stopwords)#停用词
    stops=list(stops)
    stops=stops+["n't","''",'rt','1',' ','评价','"','｀','～','\n','/','ω',')','(','_','＝','=','?','??','I','|']#可以在stop_words.txt里加入停用词，也可以在这里写入
    cutwords3 = [word for word in cutwords2 if word not in stops]
    cutwords3=[x for x in cutwords3 if x.isdigit() is False]
    cutwords3=' '.join(cutwords3)
    return cutwords3

def tfidf(texts,maxf):
    #我今天吃饭了  【我 我 我 我 今天 吃饭 吃饭，我 我 我 打球】    [1,1,2]  [0.2,0.4,0.8]
    vectorizer = CountVectorizer(decode_error="replace",max_features=maxf)#最大特征词数
    tfidftransformer = TfidfTransformer()
    # 注意在训练的时候必须用vectorizer.fit_transform、tfidftransformer.fit_transform
    # 在预测的时候必须用vectorizer.transform、tfidftransformer.transform
    vec_train = vectorizer.fit_transform(texts)#转换好的词频矩阵
    tfidf = tfidftransformer.fit_transform(vec_train)#词频矩阵转换成tfidf矩阵

    # 保存经过fit的vectorizer 与 经过fit的tfidftransformer,预测时使用
    feature_path = 'feature.pkl'
    with open(feature_path, 'wb') as fw:
        pickle.dump(vectorizer.vocabulary_, fw)

    tfidftransformer_path = 'tfidftransformer.pkl'
    with open(tfidftransformer_path, 'wb') as fw:
        pickle.dump(tfidftransformer, fw)
    print('tfidf:',tfidf.shape)#查看tfidf的shape大小
    print('*******************************************')

#获取文本的tfidf
def get_tfidf(text):
    # 加载特征
    text=get_words(text)
    feature_path = 'feature.pkl'
    loaded_vec = CountVectorizer(decode_error="replace", vocabulary=pickle.load(open(feature_path, "rb")))
    # 加载TfidfTransformer
    tfidftransformer_path = 'tfidftransformer.pkl'
    tfidftransformer = pickle.load(open(tfidftransformer_path, "rb"))
    # 测试用transform，表示测试数据，为list
    test_tfidf = tfidftransformer.transform(loaded_vec.transform([text])).toarray()
    return test_tfidf


if __name__=='__main__':
    df=pd.read_csv('../data/train.csv')
    texts=df['text'].to_list()
    print(df)
    train_text=[]
    #处理文本数据 进行分词去停用词等
    for text in tqdm(texts):
        train_text.append(get_words(text))
    tfidf(train_text,5000)