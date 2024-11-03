# coding=utf-8


class Config(object):
    def __init__(self):
        #人民日报语料库         ##数据集地址
        # self.label_file = './data/tag/PeopleDaliy_tag.txt'
        # self.train_file = './data/People_daliy/People_Daily_train.txt'
        # self.dev_file = './data/People_daliy/People_Daily_dev.txt'
        # self.test_file = './data/People_daliy/People_Daily_test.txt'
        #MASA语料库
        # self.label_file = './data/tag/MASA_tag.txt'
        # self.train_file = './data/MASA/MASA_train.txt'
        # self.dev_file   = './data/MASA/MASA_dev.txt'
        # self.test_file  = './data/MASA/MASA_test.txt'
        # Boson语料库
        # self.label_file = './data/tag/Boson_tag.txt'
        # self.train_file = './data/Boson/boson_train.txt' 
        # self.dev_file   = './data/Boson/boson_dev.txt'
        # self.test_file  = './data/Boson/boson_test.txt'
        #微博语料库
        # self.label_file = './data/tag/Weibo_tag.txt'
        # self.train_file = './data/Weibo/weiboNER_train.txt' 
        # self.dev_file   = './data/Weibo/weiboNER_dev.txt'
        # self.test_file  = './data/Weibo/weiboNER_test.txt'

        # 简历语料库
        self.label_file = './data/tag/Resume_tag.txt'
        self.train_file = './data/Resume/Resume_train.txt'
        self.dev_file   = './data/Resume/Resume_dev.txt'
        self.test_file  = './data/Resume/Resume_test.txt'

        #CHE_txt原始版本
        self.label_file = './data/tag/CHE_tag.txt'
        self.train_file = './data/CHE_TXT/CHE_traindataset.txt'
        self.dev_file = './data/CHE_TXT/CHE_devdataset.txt'
        self.test_file = './data/CHE_TXT/CHE_testdataset.txt'

        # #CHE_txt_622版本
        # self.label_file = './data/tag/CHE_tag.txt'
        # self.train_file = './data/test622txt/CHE_traindataset.txt'
        # self.dev_file = './data/test622txt/CHE_devdataset.txt'
        # self.test_file = './data/test622txt/CHE_testdataset.txt'

        # # #CHE_txt剔除俩实体
        # self.label_file = './data/tag/CHE_tag_liu.txt'
        # self.train_file = './data/CHE_TXT/C_train.txt'
        # self.dev_file = './data/CHE_TXT/C_dev.txt'
        # self.test_file = './data/CHE_TXT/C_test.txt'

        ## 细粒度
        # self.label_file = './data/tag/CHE_BMEO.txt'
        # self.train_file = './data/CHE_BMEO_TXT/CHE_traindataset.txt' 
        # self.dev_file = './data/CHE_BMEO_TXT/CHE_devdataset.txt'
        # self.test_file = './data/CHE_BMEO_TXT/CHE_testdataset.txt'

        ## CHE_v2_plus
        # self.label_file = './data/tag/CHE_tag.txt'
        # self.train_file = './data/CHE_TXT_V2/CHE_traindataset.txt'
        # self.dev_file = './data/CHE_TXT_V2/CHE_devdataset.txt'
        # self.test_file = './data/CHE_TXT_V2/CHE_testdataset.txt'


        ## 预训练模型修改============================================================================
        # self.vocab = './data/ernie-base-chinese/vocab.txt'  ##ernie1.0
        # self.ernie_path = './data/ernie-base-chinese'  ##ernie1.0
        # self.bert_path = './data/bert-base-chinese'  ##bert
        # self.vocab = './data/bert-base-chinese/vocab.txt'##bert
        # self.ernie3_path = './data/ERNIE3.0-base-chinese'  ##ernie3.0
        # self.vocab = './data/ERNIE3.0-base-chinese/vocab.txt'  ##ernie3.0
        self.ernie3_path = './data/robert'  ##ernie3.0
        self.vocab = './data/robert/vocab.txt'  ##ernie3.0
        ## 预训练模型修改============================================================================

        self.max_length = 100
        self.use_cuda = True
        self.use_llm_embedding = False
        self.use_attention = True
        self.gpu = 0
        self.batch_size = 50


        ## 预训练模型嵌入维度修改============================================================================
        self.ernie_embedding = 768  # ernie1.0
        #self.ernie_embedding = 1024
        # self.bert_embedding = 768#bert
        ##self.ernie3_embedding = 768
        ## 预训练模型嵌入维度修改============================================================================

        self.rnn_hidden = 500
        self.num_heads = 8

        self.dropout1 = 0.5
        self.dropout_ratio = 0.5
        self.rnn_layer = 1
        self.lr = 5e-5
        self.lr_decay = 0.00001
        self.weight_decay = 0.00005
        self.checkpoint = 'result/'
        self.optim = 'Adam'
        ## 是否加载模型   模型与路径
        self.load_model =  False
        # self.load_path = 'PeopleDaily-9718'
        # self.load_path = 'MASA-98386'
        # self.load_path = 'Boson-9059'
        # self.load_path = 'Weibo-8205'
        # self.load_path = 'resume-0312'
        self.load_path = 'CHE_TXT-2392'
        ## 训练轮次
        self.base_epoch = 50
        self.require_improvement = 1000


        self.XH_Train_code = "./data/Embedding_XH_code/weibo_train_embedding.pt"
        self.XH_Dev_code = "./data/Embedding_XH_code/weibo_dev_embedding.pt"
        self.XH_Test_code = "./data/Embedding_XH_code/weibo_test_embedding.pt"

    def update(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __str__(self):

        return '\n'.join(['%s:%s' % item for item in self.__dict__.items()])

