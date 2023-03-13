'''此处是评估代码，用来验证我们推荐的准确率'''

import pandas as pd
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from train import get_tfidf
import pickle
from sklearn.metrics import classification_report

#传递查询数据文本和数据库文本，返回其向量和标签用于计算相似度
def get_tensor(source,target):
    s_list=[]#承接数据库向量和标签
    t_list=[]#承接查询向量和标签
    for x in tqdm(range(len(source))):
        st=get_tfidf(source.iloc[x,1])#向量
        sl=source.iloc[x,2]#标签
        s_list.append((st,sl,source.iloc[x,1]))
    #将计算完文本向量的数据库保存
    with open('souce.pkl','wb')as f:
        pickle.dump(s_list,f)

    for x in tqdm(range(len(target))):
        tt=get_tfidf(target.iloc[x,1])#向量
        tl=target.iloc[x,2]#标签
        t_list.append((tt,tl,target.iloc[x,1]))
    #将计算完文本向量的查询文件保存
    with open('target.pkl','wb')as f:
        pickle.dump(t_list,f)

def main():
    with open('souce.pkl', 'rb') as f:
        source = pickle.load(f)

    with open('target.pkl', 'rb') as f:
        target = pickle.load(f)

    y_true=[]#查询文件的真实标签
    y_pre=[]#推荐文件的标签
    for t in tqdm(target,'正在计算准确率'):
        tn=t[0]#查询文件的向量
        tlabel=t[1]#查询文件的标签
        y_true.append(tlabel)

        sim_list=[]#承接相似度结果和标签
        for s in source:
            sn=s[0]#数据库文件的向量
            slabel=s[1]#数据库文件的标签
            sim = cosine_similarity(tn,sn)[0][0]#查询文件和数据库文件的相似度
            sim_list.append((sim,slabel))
        sim_list = sorted(sim_list, key=lambda kv: kv[0], reverse=True)[:1]#找出最相似的文章标签
        y_pre.append(sim_list[0][1])
    print(classification_report(y_true,y_pre))


if __name__=='__main__':
    df = pd.read_csv('../data/test.csv')
    '''在我们的测试集中抽出10%的数据作为查询数据，剩下的为待查询数据库，
    将每个查询数据查找最相似的数据库数据，如果类别一致为查询推荐成功'''
    #650 5900#  体育 体育  1  0
    source,target=train_test_split(df,test_size=0.1)
    #get_tensor(source,target)
    main()
