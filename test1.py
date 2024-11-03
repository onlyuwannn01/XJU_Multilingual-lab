# from gensim import corpora
# import matplotlib.pyplot as plt
# import matplotlib
# import warnings

# from gensim.models.coherencemodel import CoherenceModel
# from gensim.models.ldamodel import LdaModel

# warnings.filterwarnings('ignore')  # 忽略警告信息以提高清晰度

# PATH1 = "LLM_fenci.txt"

# # 读取内容，一行行地
# file_object2 = open(PATH1, encoding='utf-8', errors='ignore').read().split('\n')
# data_set = []  # 存储分词结果的列表
# for line in file_object2:
#     seg_list = line.split()
#     data_set.append(seg_list)

# # 构建词典和语料库
# dictionary = corpora.Dictionary(data_set)
# corpus = [dictionary.doc2bow(text) for text in data_set]

# # 计算困惑度
# def perplexity(num_topics):
#     print(f'num_topics:{num_topics}')
#     ldamodel = LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=30)
#     print(ldamodel.print_topics(num_topics=num_topics, num_words=15))
#     print(ldamodel.log_perplexity(corpus))
#     return ldamodel.log_perplexity(corpus)


# # 计算一致性
# def coherence(num_topics):
#     print(f'num_topics:{num_topics}')
#     ldamodel = LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=30, random_state=1)
#     print(ldamodel.print_topics(num_topics=num_topics, num_words=10))
#     ldacm = CoherenceModel(model=ldamodel, texts=data_set, dictionary=dictionary, coherence='c_v')
#     print(ldacm.get_coherence())
#     return ldacm.get_coherence()

# if __name__ == '__main__':
#     x = range(1, 20)  # 主题数范围
#     perplexities = [perplexity(i) for i in x]  # 计算每个主题数的困惑度
#     coherences = [coherence(i) for i in x]  # 计算每个主题数的一致性得分

#     # 绘制困惑度图
#     plt.figure(figsize=(10, 5))
#     plt.plot(list(x), perplexities, label='Perplexity')
#     plt.xlabel('Number of Topics')
#     plt.ylabel('Perplexity')
#     plt.title('Perplexity of LDA Models_zky_v2')
#     plt.legend()
#     plt.grid(True)
#     plt.xticks(list(x))
#     plt.show()

#     # 绘制一致性得分图
#     plt.figure(figsize=(10, 5))
#     plt.plot(list(x), coherences, label='Coherence', color='red')
#     plt.xlabel('Number of Topics')
#     plt.ylabel('Coherence Score')
#     plt.title('Coherence Score of LDA Models_zky_v2')
#     plt.legend()
#     plt.grid(True)
#     plt.xticks(list(x))
#     plt.show()
import os
from gensim import corpora
from gensim.models.ldamodel import LdaModel
from gensim.models.coherencemodel import CoherenceModel
import matplotlib.pyplot as plt

def preprocess(path):
    with open(path, encoding='utf-8', errors='ignore') as file:
        data_set = [line.split() for line in file.read().split('\n') if line]
    return data_set

def build_corpus(data_set):
    dictionary = corpora.Dictionary(data_set)
    corpus = [dictionary.doc2bow(text) for text in data_set]
    return dictionary, corpus

def evaluate_model(corpus, dictionary, data_set, num_topics, passes=30, random_state=1):
    ldamodel = LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=passes, random_state=random_state)
    perplexity = ldamodel.log_perplexity(corpus)
    coherence_model = CoherenceModel(model=ldamodel, texts=data_set, dictionary=dictionary, coherence='c_v')
    coherence = coherence_model.get_coherence()
    return perplexity, coherence

# 修改绘图的函数以支持多个曲线，并保存图形到指定文件夹
def plot_evaluation(x, ys,  title, ylabel, colors, folder_path, file_name):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    plt.figure(figsize=(10, 5))
    for y, color in zip(ys,  colors):
        plt.plot(x, y, color=color)
    plt.xlabel('Number of Topics')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.xticks(list(x))
    plt.savefig(os.path.join(folder_path, file_name))
    plt.close()  # 关闭图形，避免在notebook中显示

if __name__ == '__main__':
    folder_path = 'picture'  # 图片保存的文件夹
    paths = ["LLM_fenci.txt"]
    colors = ['red', 'blue']
    #labels = ['jieba_full_1', 'jieba_full_2']
    num_topics_range = range(1, 21)  # 主题数范围

    perplexities_list = []
    coherences_list = []

    for path in paths:
        print(f"Processing file: {path}")
        data_set = preprocess(path)
        dictionary, corpus = build_corpus(data_set)
        perplexities = []
        coherences = []

        for num_topics in num_topics_range:
            print(f"Evaluating model with {num_topics} topics...")
            perplexity, coherence = evaluate_model(corpus, dictionary, data_set, num_topics=num_topics)
            perplexities.append(perplexity)
            coherences.append(coherence)

        perplexities_list.append(perplexities)
        coherences_list.append(coherences)
        print(f"Completed processing file: {path}\n")

    # 绘制困惑度和一致性曲线，并保存到指定文件夹
    plot_evaluation(num_topics_range, perplexities_list,  'Perplexity of LDA Models', 'Perplexity', colors, folder_path, 'perplexity.png')
    plot_evaluation(num_topics_range, coherences_list,  'Coherence Score of LDA Models', 'Coherence Score', colors, folder_path, 'coherence.png')