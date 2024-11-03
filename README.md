# RoBERTa-BiLSTM-Attention-CRF-Pytorch-Chinese_NER
基于RoBERTa的中文NER
# RoBERTa-BiLSTM-Attention-CRF
## 模型架构
![RoBERTa-BiLSTM-Attention-CRF.svg]
### RoBERTa层
采用预训练语言模型ERNIE对输入的文本数据进行向量化表示
### BiLSTM
通过双向循环神经网络(BiLSTM)进行特征提取提取编码得到一个得分矩阵<br/>
LSTM细胞结构：
![lstm.jpg](https://p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/21f4910840af4397a28f1bd588dd3130~tplv-k3u1fbpfcp-watermark.image?)
计算如式：
![遗忘门.jpg](https://p9-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/f6bc4137303d44e1813eddd8797a5d23~tplv-k3u1fbpfcp-watermark.image?)
![输入门it.jpg](https://p1-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/79f49d32fdc349fb8d9bb35f22bd4e0f~tplv-k3u1fbpfcp-watermark.image?)
![候选细胞状态.jpg](https://p6-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/29ee9df086944c2ba4313c1cc1de9cb4~tplv-k3u1fbpfcp-watermark.image?)
![输出门ot.jpg](https://p9-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/44320d39c1b949d6aed8a4a213782dbb~tplv-k3u1fbpfcp-watermark.image?)
![隐层状态ht.jpg](https://p6-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/d0a2bd355e4a4de08ad01774bfdbb9b5~tplv-k3u1fbpfcp-watermark.image?)
BiLSTM细胞结构:
![biLSTM(1).jpg](https://p6-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/a4e462fdf3a74b8aa6b5d358395956e7~tplv-k3u1fbpfcp-watermark.image?)
### CRF
通过条件随机场(CRF)进行解码，再用维特比算法得到概率最大的一组标签作为算法的输出<br/>
计算如式：
![swy.jpg](https://p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/ba07307054ff4e71a0b7ef427550a7d9~tplv-k3u1fbpfcp-watermark.image?)
![pxy.jpg](https://p1-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/e933d5a33c1d4e73aff86aa05f8e697a~tplv-k3u1fbpfcp-watermark.image?)
![log.jpg](https://p9-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/26b572ddb9cc4ad4aac6658bae99700a~tplv-k3u1fbpfcp-watermark.image?)
![Y_.jpg](https://p9-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/a177de49aedc44ba9410595b041cf095~tplv-k3u1fbpfcp-watermark.image?)
# BERT-BiLSTM-CRF
## 模型架构
![Bert_bilstm-crf.svg](https://p9-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/4b0366c0644e4db3a8c853fabcaf6a6c~tplv-k3u1fbpfcp-watermark.image?)
除了向量化采用BERT，其他与前者相同

# 实验
## 环境
使用此平台[openbayes](https://openbayes.com/)的容器可以省去搭环境的麻烦，数据会保存
每次需要安装依赖
> pip3 install pytorch-crf -i https://pypi.mirrors.ustc.edu.cn/simple/  
> pip3 install transformers -i https://pypi.mirrors.ustc.edu.cn/simple/  
> pip3 install fire -i https://pypi.mirrors.ustc.edu.cn/simple/  
> pip3 install seqeval -i https://pypi.mirrors.ustc.edu.cn/simple/

