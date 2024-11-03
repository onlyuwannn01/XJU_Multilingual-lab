# coding=utf-8
import torch
import os
import datetime
import unicodedata
from torch.utils.data import TensorDataset
from datetime import timedelta
import time
from torch.utils.data import Dataset
class InputFeatures(object):
    def __init__(self, input_id, label_id, input_mask):
        self.input_id = input_id
        self.label_id = label_id
        self.input_mask = input_mask


def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    vocab = {}
    index = 0
    with open(vocab_file, "r", encoding="utf-8") as reader:
        while True:
            token = reader.readline()
            if not token:
                break
            token = token.strip()
            vocab[token] = index
            index += 1
    return vocab


def read_corpus(path, max_length, label_dic, vocab):
    """
    :param path:数据文件路径
    :param max_length: 最大长度
    :param label_dic: 标签字典
    :return:
    """
    file = open(path, encoding='utf-8')
    content = file.readlines()
    file.close()
    result = [] 
    tokens = []
    label = []
    
    for line in content:
        # 读取一行
        if line != '\n':
            word, tag = line.strip('\n').split()
            tokens.append(word)
            label.append(tag)
        #获得一句话
        else:
            if len(tokens) > max_length - 2:
                tokens = tokens[0:(max_length - 2)]
                label = label[0:(max_length - 2)]
            tokens_f = ['[CLS]'] + tokens + ['[SEP]']
            #label_f = ["<start>"] + label + ['<eos>'] 
            label_f = ["<s>"] + label + ['</s>']#===============================================robert
            input_ids = [int(vocab[i]) if i in vocab else int(vocab['[UNK]']) for i in tokens_f]
            label_ids = [label_dic[i] for i in label_f]
            input_mask = [1] * len(input_ids)
            while len(input_ids) < max_length:
                input_ids.append(0)
                input_mask.append(0)
                label_ids.append(label_dic['<pad>'])
            assert len(input_ids) == max_length
            assert len(input_mask) == max_length
            assert len(label_ids) == max_length
            feature = InputFeatures(input_id=input_ids, input_mask=input_mask, label_id=label_ids)
            result.append(feature)
            tokens = []
            label = []
    return result

def build_dataset(data):
    """
    生成数据集
    """
    input_ids = torch.LongTensor([temp.input_id for temp in data])
    input_masks = torch.LongTensor([temp.input_mask for temp in data])
    label_ids = torch.LongTensor([temp.label_id for temp in data])
    dataset = TensorDataset(input_ids, input_masks, label_ids)
    return dataset

def save_model(model, epoch, path='result/', **kwargs):
    """
    默认保留所有模型
    :param model: 模型
    :param path: 保存路径
    :param loss: 校验损失
    :param last_loss: 最佳epoch损失
    :param kwargs: every_epoch or best_epoch
    :return:
    """
    if not os.path.exists(path):
        os.mkdir(path)
    if kwargs.get('name', None) is None:
        cur_time = datetime.datetime.now().strftime('%Y-%m-%d#%H时%M分%S秒')
        # name = 'PeopleDaily-ERNIE-BiLSTM-CRF' +cur_time + '--epoch=%d' % epoch
        # name = 'MASA-ERNIE-BiLSTM-CRF' +cur_time + '--epoch=%d' % epoch
        # name = 'Boson-ERNIE-BiLSTM-CRF' +cur_time + '--epoch=%d' % epoch
        # name = 'Weibo-ERNIE-BiLSTM-CRF' +cur_time + '--epoch=%d' % epoch
        # name = 'Resume-ERNIE-BiLSTM-CRF' + cur_time + '--epoch=%d' % epoch



        # name = 'PeopleDaily-ERNIE- BiGRU-CRF' +cur_time + '--epoch=%d' % epoch
        # name = 'MASA-ERNIE- BiGRU-CRF' +cur_time + '--epoch=%d' % epoch
        # name = 'Boson-ERNIE- BiGRU-CRF' +cur_time + '--epoch=%d' % epoch
        # name = 'Weibo-ERNIE- BiGRU-CRF' +cur_time + '--epoch=%d' % epoch
        # name = 'Resume-ERNIE- BiGRU-CRF' + cur_time + '--epoch=%d' % epoch

        # name = 'PeopleDaily-ERNIE3.0- BiGRU-CRF' +cur_time + '--epoch=%d' % epoch
        # name = 'MASA-ERNIE3.0- BiGRU-CRF' +cur_time + '--epoch=%d' % epoch
        # name = 'Boson-ERNIE3.0- BiGRU-CRF' +cur_time + '--epoch=%d' % epoch
        # name = 'Weibo-ERNIE3.0- BiGRU-CRF' +cur_time + '--epoch=%d' % epoch
        # name = 'Resume-ERNIE3.0- BiGRU-CRF' + cur_time + '--epoch=%d' % epoch

        # name = 'PeopleDaily-ERNIE3.0- BiLSTM-CRF' +cur_time + '--epoch=%d' % epoch
        # name = 'MASA-ERNIE3.0- BiLSTM-CRF' +cur_time + '--epoch=%d' % epoch
        # name = 'Boson-ERNIE3.0- BiLSTM-CRF' +cur_time + '--epoch=%d' % epoch
        # name = 'Weibo+ERNIE3.0- BiLSTM-CRF' +cur_time + '--epoch=%d' % epoch
        # name = 'Resume-ERNIE3.0- BiLSTM-CRF' + cur_time + '--epoch=%d' % epoch

        # name = 'CHE-ERNIE3.0- BiLSTM-CRF' +cur_time + '--epoch=%d' % epoch
        # name = 'CHE-ERNIE3.0- BiGRU-CRF' +cur_time + '--epoch=%d' % epoch

        #name = 'CHE-BMEO-Ernie- Bigru-CRF_1945' +cur_time + '--epoch=%d' % epoch

        # name = 'CHE-ERNIE3.0- BiLSTM-MHATT-CRF' +cur_time + '--epoch=%d' % epoch
        # name = 'CHE-ERNIE3.0- BiGRU-MHATT-CRF' +cur_time + '--epoch=%d' % epoch
        name = 'CHE-Bio-robert- Bigru-CRF_999' +cur_time + '--epoch=%d' % epoch
       
        full_name = os.path.join(path, name)
        torch.save(model.state_dict(), full_name)
        print('Saved model at epoch {} successfully'.format(epoch))
        with open('{}/checkpoint'.format(path), 'w') as file:
            file.write(name)
            print('Write to checkpoint')


def load_model(model, path='result/', **kwargs):
    if kwargs.get('name', None) is None:
        with open('{}/checkpoint'.format(path)) as file:
            content = file.read().strip()
            name = os.path.join(path, content)
    else:
        name=kwargs['name']
        name = os.path.join(path,name)
    model.load_state_dict(torch.load(name, map_location=lambda storage, loc: storage))
    print('load model {} successfully'.format(name))
    return model

def get_time_diff(start_time):
    end_time = time.time()
    time_diff = end_time - start_time
    return timedelta(seconds=int(round(time_diff)))


    
def read_pt_file(file_path):  ##改===============================
    tensor = torch.load(file_path)
    return tensor





class NerDataset(Dataset):  ##改===============================
    """
    一个自定义的数据集类，用于命名实体识别（NER）任务。
    """

    def __init__(self, data,data_embedding):
        """
        初始化NerDataset类的实例。

        参数:
        - data: 包含多个具有input_id, input_mask, label_id属性的对象的列表。
        """
        # 使用列表推导式提取input_ids, input_masks, 和label_ids，并转换为Tensor。
        self.input_ids = torch.LongTensor([temp.input_id for temp in data])
        self.input_masks = torch.LongTensor([temp.input_mask for temp in data])
        self.label_ids = torch.LongTensor([temp.label_id for temp in data])
        self.data_embedding_ids =torch.Tensor(data_embedding)
    def __getitem__(self, index):
        """
        根据索引获取数据集中的单个样本。

        参数:
        - index: 要获取的样本的索引。

        返回值:
        - 一个包含input_id, input_mask, 和label_id的字典。
        """
        return {
            'input_ids': self.input_ids[index],
            'input_masks': self.input_masks[index],
            'label_ids': self.label_ids[index],
            'data_embedding_ids':self.data_embedding_ids[index]
        }

    def __len__(self):
        """
        获取数据集中样本的总数。

        返回值:
        - 数据集中样本的数量。
        """
        return len(self.input_ids)
