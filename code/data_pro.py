import pandas as pd
import json
from sklearn.model_selection import train_test_split
'''这一步是为了划分训练验证测试数据集'''
df=pd.read_csv('data/data.csv')

labels=set(df['label'].to_list())
print(labels)
#获取每个文本标签对应的数字标签，例如科技对应0 体育对应1
label_index=[(x,index) for index,x in enumerate(labels)]
index_label=[(index,x) for index,x in enumerate(labels)]
label_index=dict(label_index)
index_label=dict(index_label)
print(label_index)
print(index_label)
#保存标签对应
with open('data/label_index.json','w',encoding='utf-8')as f:
    json.dump(label_index,f,ensure_ascii=False)
with open('data/index_label.json','w',encoding='utf-8')as f:
    json.dump(index_label,f,ensure_ascii=False)

newlabel=[]
str_num=0
for x in range(len(df)):
    l=df.iloc[x,0]
    str_num=len(df.iloc[x,1])+str_num
    newl=label_index[l]
    newlabel.append(newl)
df['newlabel']=newlabel
print('平均新闻长度：',str_num/len(df))
#划分训练验证测试 训练集0.8  测试验证0.1
train,_=train_test_split(df,test_size=0.2)
val,test=train_test_split(_,test_size=0.5)
# print(train)
# print(test)
# print(val)
train.to_csv('data/train.csv',index=None)
val.to_csv('data/val.csv',index=None)
test.to_csv('data/test.csv',index=None)