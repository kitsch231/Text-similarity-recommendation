# coding=utf-8
'''此处是预测或者说推荐代码，通过我们的建模和数据库的建立，输入一条新闻获得推荐的结果'''

import pandas as pd
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from train import get_tfidf
import pickle
from sklearn.metrics import classification_report
import json
def main(tn):
    with open('souce.pkl', 'rb') as f:
        source = pickle.load(f)

    with open('../data/index_label.json', 'r',encoding='utf-8') as f:
        index_label=json.load(f)


    sim_list=[]#承接相似度结果和标签
    for s in source:
        sn=s[0]#数据库文件的向量
        slabel=s[1]#数据库文件的标签
        stext=s[2]#数据库文件的文本
        sim = cosine_similarity(tn,sn)[0][0]#查询文件和数据库文件的相似度
        sim_list.append((sim,slabel,stext))
    sim_list = sorted(sim_list, key=lambda kv: kv[0], reverse=True)[:10]#找出最相似的5个文章
    #输出推荐相似度 标签 新闻
    for index,s in enumerate(sim_list):
        print('推荐第{}条新闻相似度:{}'.format(index,s[0]))
        print('推荐第{}条新闻相类别:{}'.format(index, index_label[str(s[1])]))
        print('推荐第{}条新闻相内容:{}'.format(index, s[2]))
        print('******************************************')




if __name__=='__main__':
    text='组图豹纹围巾露肩长裙 细节搭配穿出性感身材导语：曾记得一个传奇型设计师说道：“你要看一个人是否会穿着，懂得品味，懂得时尚？一定要看她的细节，她拿什么包，她穿什么鞋，她怎么佩戴她的首饰。”正所谓细节成就一切。这个Look其实很有Carrie的影子！即使用宽大的豹纹来搭配，也没有给矮小的身材带来灾难，内里搭配的民族风长衫有做旧的效果，十分随意复古，用满是气场的豹纹方巾做修饰，增加了时尚度。而红袜搭短靴也很容易抓人眼球。一件T和长裙穿出这样的气质，着实很考验搭配的功力。灰T露肩穿，衣角扎进裙，但是有打斜的处理，给本来单调规矩的单品增添了层次感，肩头小露的性感，估计是谁都能第一眼注意到。埃及艳后的发型，搭配皮衣简直酷毙了！都是属于硬朗个性风，自然融合出凌厉的气质，红色丝袜给暗色Look注入了火辣性感的味道。部落风的长裤用卡其色风衣搭，显现出优雅大气的质感，暗哑的铜质流苏项链巧妙点缀，增加上身的厚实度，摩登时髦。'
    tn=get_tfidf(text)#本新闻文本的tfidf向量
    main(tn)
