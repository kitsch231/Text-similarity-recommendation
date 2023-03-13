# coding=utf-8
'''此处是预测或者说推荐代码，通过我们的建模和数据库的建立，输入一条新闻获得推荐的结果'''

import pandas as pd
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
import pickle
from sklearn.metrics import classification_report
import json
from Config import Config
from models import Mynet
import torch
from transformers import BertTokenizer

config = Config()
model=Mynet(config).to(config.device)
model.load_state_dict(torch.load(config.save_path))
tokenizer = BertTokenizer.from_pretrained(config.bert_name)

#通过bert模型获取特征向量
def get_t(text):
    model.eval()
    with torch.no_grad():
        text=tokenizer(text=text, add_special_tokens=True,
                  max_length=config.pad_size,  # 最大句子长度
                  padding='max_length',  # 补零到最大长度
                  truncation=True)

        input_id= torch.tensor(text['input_ids'], dtype=torch.long).unsqueeze(0)
        attention_mask = torch.tensor(text['attention_mask'], dtype=torch.long).unsqueeze(0)#可用可不用
        input=(input_id.to(config.device),attention_mask.to(config.device))
        fea, outputs = model(input)
        fea=fea.data.cpu().numpy()
        #print(fea.shape)
    return fea

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
    text='留学生口述：国外要拿学位其实很不容易与国内大学“严进宽出”相反，国外大学实行“宽进严出”的政策，绝大部分海外高校实行淘汰制，留学生仿佛套着紧箍咒，每学年都面临着被“扫地出门”的危险，顺利拿到毕业文凭相当不容易。中国留学生在国外学习成绩都不错？不少留学生告诉记者，这个“不错”印象的背后，有血有泪有辛酸。喷鼻血也要复习小宇 墨尔本留学时间：3年小宇介绍说，学文科的学生考试占整门课的30%-40%，平时成绩占大部分，通常是写两篇论文和在线课堂讨论。上课分为两种类型：lecture和tutorial。lecture是100人左右的课，不用签到，tutorial只有10-20人，要记录出勤率。一个学期13周，第一周是这门课的介绍，最后一周是考前预习，然后有1个月备考。一个学期只有4门课，学得很精。小宇告诉记者，一般在澳洲大学里考高分的中国学生都是学习数学、会计、计算机等理工科的，但是对语言要求高的专业，如传媒、教育，因为要学习西方历史、哲学、文学等，中国学生很难拿高分。在国外读书不像国内可以临时抱佛脚，考试的内容贯穿一个学期。每门课的及格率大概在50%，没有补考，只能交1万多元人民币重修。重修两次不及格就会被开除。一说到考试，小宇就想起一段难忘的经历：“2008年11月的考试，墨尔本45℃，我住的地方没空调。白天还可以去图书馆，可是到了晚上，大家都挤在24小时麦当劳里复习。晚上我热得睡不着，就在车里开3个小时空调眯一会儿，可还是顶不住，热到流鼻血也只能咬牙坚持着把考试过了。”一半学生不能毕业范范 鹿特丹留学时间：2年范范当初高考失利，被父母送到了荷兰。因为实行淘汰制，荷兰大学的毕业压力很大。一门课满分10分，8—10分属于优秀，7分已经是好学生了。知名大学的教师出卷子，预计及格率仅是30%，一年内不及格的科目不得超过3门，否则要被开除。到了第二年，如果还有第一年没过的科目，同样会被开除。普通大学毕业率也只有50%—60%左右。范范告诉记者，每年都有不少留学生自动退学。“如果实在读不下去，不妨转到稍差的学校，一模一样的课程可以免修，其他的课程必须重读。”虽然很辛苦，但范范觉得收获不少：“这些年我成长了不少，所有事情都靠自觉。很多留学生来了以后都判若两人了，学习特别刻苦。”做好每个小测验徐青 温哥华留学时间：5年“多看书、多上课、多记笔记。”这是徐青首先给留学生的建议。现在徐青已经完成金融学专业的课程，正在办理加拿大移民。在国内，就一个期末考试，而在加拿大，期末成绩基本由平时一点点累加的。期末总分通常是由许多部分组成，比如平时的QUIZ(随堂测验)、TEST(小测验)、ASSIGNMENT(作业)和PAPER(论文)，还有EXAM(期终考试)。想取得好成绩就不要走捷径，低年级的同学要认真对待每堂课、认真做笔记、做好回家作业和每一个单元的小测验。这样考试的时候即使成绩不是很高，期末平均成绩也还不错。到了高年级的课程，论文、团队合作演讲、小组论文这些内容考验的是综合素质，要对专业知识融会贯通。所以，需要平时和其他同学多交流，一起做作业，考试前还可以请教下教授。“有一点很重要：论文决不允许抄袭！”徐青强调说。'
    tn=get_t(text)#本新闻文本的tfidf向量
    main(tn)
