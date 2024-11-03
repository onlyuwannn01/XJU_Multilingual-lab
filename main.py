# coding=utf-8
import torch,gc
import torch.nn as nn
from torch.autograd import Variable
from config import Config
from model import ERNIE_LSTM_CRF, ernie_gru_crf
import torch.optim as optim
from utils import load_vocab, read_corpus, load_model, save_model, build_dataset, get_time_diff,read_pt_file,NerDataset
from torch.utils.data import DataLoader
import fire
import warnings
import time
import random
import numpy as np
import os
import datetime
from seqeval.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore', category=UserWarning)
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

def set_seed(seed_value):
    # 设置Python的随机数种子
    random.seed(seed_value)
    np.random.seed(seed_value)

    # 设置PyTorch的随机数种子
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def train(**kwargs):
    config = Config()
    config.update(**kwargs)
    print('当前设置为:\n', config)
    if config.use_cuda:
        torch.cuda.set_device(config.gpu)
    print('loading corpus')
    vocab = load_vocab(config.vocab) # {token: index}
    label_dic = load_vocab(config.label_file) # {tag: index}
    id2tag = {label_dic[tag]: tag for tag in label_dic.keys()}
    tagset_size = len(label_dic)
    train_data = read_corpus(config.train_file, max_length=config.max_length, label_dic=label_dic, vocab=vocab)
    dev_data = read_corpus(config.dev_file, max_length=config.max_length, label_dic=label_dic, vocab=vocab)
    test_data = read_corpus(config.test_file, max_length=config.max_length, label_dic=label_dic, vocab=vocab)


    #train_dataset = build_dataset(train_data) ## 原始
    if config.use_llm_embedding:
        train_tensor = read_pt_file(config.XH_Train_code)    ##改===============================
        train_dataset =NerDataset(train_data,train_tensor)  ##改===============================
        train_loader = DataLoader(train_dataset, shuffle=True, batch_size=config.batch_size)

        dev_tensor = read_pt_file(config.XH_Dev_code) ##改===============================
        dev_dataset = NerDataset(dev_data,dev_tensor) ##改===============================
        dev_loader = DataLoader(dev_dataset, shuffle=True, batch_size=config.batch_size)

        test_tensor = read_pt_file(config.XH_Test_code) ##改===============================
        test_dataset = NerDataset(test_data,test_tensor) ##改===============================
        test_loader = DataLoader(test_dataset, shuffle=True, batch_size=config.batch_size)
    else:
        train_dataset = build_dataset(train_data)
        train_loader = DataLoader(train_dataset, shuffle=True, batch_size=config.batch_size)

        dev_dataset = build_dataset(dev_data)
        dev_loader = DataLoader(dev_dataset, shuffle=True, batch_size=config.batch_size)

        test_dataset = build_dataset(test_data)
        test_loader = DataLoader(test_dataset, shuffle=True, batch_size=config.batch_size)

    #model = ERNIE_LSTM_CRF(config.ernie3_path, tagset_size, config.ernie_embedding, config.rnn_hidden, config.rnn_layer, dropout_ratio=config.dropout_ratio, dropout1=config.dropout1, use_cuda=config.use_cuda,use_llm=config.use_llm_embedding)
    model = ernie_gru_crf.ERNIE_GRU_CRF(config.ernie3_path, tagset_size, config.ernie_embedding, config.rnn_hidden, config.rnn_layer, dropout_ratio=config.dropout_ratio, dropout1=config.dropout1, use_cuda=config.use_cuda,use_llm=config.use_llm_embedding)
    
    #model = ernie_gru_crf.ERNIE_GRU_CRF(config.ernie3_path, tagset_size, config.ernie_embedding, config.rnn_hidden, config.rnn_layer, dropout_ratio=config.dropout_ratio, dropout1=config.dropout1, use_cuda=config.use_cuda)
    if config.load_model:
        assert config.load_path is not None
        model = load_model(model, name=config.load_path)
    if config.use_cuda:
        model.cuda()

    losses = []
    accuracies = []
    steps = []

    optimizer = getattr(optim, config.optim)
    optimizer = optimizer(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=1) ## 注意==========================
    model.train() # 设置成训练模式
    step = 0
    eval_loss = float('inf')
    last_improved = 0 # 记录上一次更新的step值
    flag = False
    for epoch in range(config.base_epoch):
        true = []
        pred = []
        for i, batch in enumerate(train_loader):
            step += 1
            model.zero_grad()
            if config.use_llm_embedding:
            ##改===============================

                inputs, masks, tags,embedding = batch['input_ids'].to(DEVICE),batch['input_masks'].to(DEVICE),batch['label_ids'].to(DEVICE),batch['data_embedding_ids'].to(DEVICE)
                feats = model(inputs, embedding, masks)
            else:
                inputs, masks, tags = batch
                inputs, masks, tags = Variable(inputs), Variable(masks), Variable(tags)
                if config.use_cuda:
                    inputs, masks, tags = inputs.cuda(), masks.cuda(), tags.cuda()
                feats = model(inputs,  masks)
            loss = model.loss(feats, masks.byte(), tags)
            loss.backward()
            optimizer.step()

            best_train = model.crf.decode(feats,masks.byte())
            pred.extend( [ t for t in best_train])
            true.extend( [[ x for x in t.tolist() if x != 0] for t in tags])

            if step % 5 == 0:
                true = [[id2tag[y] for y in x] for x in true]
                pred = [[id2tag[y] for y in x] for x in pred]
                acc = accuracy_score(true, pred)
                #print('step: {} |  epoch: {}|  loss: {}  |acc:{}'.format(step, epoch, loss.item(), acc))
                print('step: {:5} | epoch: {:4} | loss: {:12.6f} | acc: {:.6f}'.format(step, epoch, loss.item(), acc))
                losses.append(loss.item())
                accuracies.append(acc)
                steps.append(step)
                true = []
                pred = []
                #print('step: {} |  epoch: {}|  loss: {}'.format(step, epoch, loss.item()))
            if step % 50 == 0:
                ## 验证
                f1, dev_loss = dev(model, dev_loader, config, id2tag, test=False) # 保存模型
                #scheduler.step(f1)  ## 注意==========================
                if dev_loss < eval_loss:
                    eval_loss = dev_loss
                    save_model(model, epoch)
                    last_improved = step
                    improve = '*'
                else:
                    improve = ''
                print('eval  epoch: {}|  f1_score: {}|  loss: {}|   {}'.format(epoch, f1, dev_loss, improve))
            if step - last_improved > config.require_improvement: # early stop
                print('No optimization for a long time, auto-stopping...')
                flag = True
                save_model(model, epoch)
                break
        if flag:
            break




    def set_xticks(steps):
        xticks = np.arange(0, steps[-1] + 100, 100)
        xticks = xticks[xticks <= ((steps[-1] // 100 + 1) * 100)]
        return xticks

# 确保输出文件夹存在
    output_folder = 'path_to_your_folder'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 绘制并保存损失图
    plt.figure(figsize=(8, 6))  # 设置图形的大小
    plt.plot(steps, losses, label='Loss', color='red')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    #plt.grid(True)
    plt.xticks(set_xticks(steps))
    plt.savefig(os.path.join(output_folder, 'loss_1950622.png'), dpi=300)  # 保存为高清图片
    plt.close()  # 关闭图形，以便开始下一个

    # 绘制并保存准确率图
    plt.figure(figsize=(8, 6))  # 设置图形的大小
    plt.plot(steps, accuracies, label='Accuracy', color='blue')
    plt.xlabel('Step')
    plt.ylabel('Accuracy')
    plt.title('Training Accuracy')
    plt.legend()
    #plt.grid(True)
    plt.xticks(set_xticks(steps))
    plt.savefig(os.path.join(output_folder, 'acc_1950622.png'), dpi=300)  # 保存为高清图片
    plt.close()  # 关闭图形，完成绘制




    test(model, test_loader, config, id2tag)


def dev(model, dev_loader, config, id2tag, test=False):
    model.eval()
    eval_loss = 0
    true = []
    pred = []
    with torch.no_grad():
        for i, batch in enumerate(dev_loader):
            if test: # 测试时间过长，打印信心可以看到测试进度
                print('处理测试集数据第' + str(i * config.batch_size) + '到第' + str((i+1) * config.batch_size) + '条...')
            if config.use_llm_embedding:
                inputs, masks, tags,embedding = batch['input_ids'].to(DEVICE),batch['input_masks'].to(DEVICE),batch['label_ids'].to(DEVICE),batch['data_embedding_ids'].to(DEVICE)

                feats = model(inputs,embedding,masks)
            else:
                inputs, masks, tags = batch
                inputs, masks, tags = Variable(inputs), Variable(masks), Variable(tags)
                if config.use_cuda:
                    inputs, masks, tags = inputs.cuda(), masks.cuda(), tags.cuda()
                feats = model(inputs,  masks)
            #feats = model(inputs, masks)
            # 此处使用维特比算法解码
            best_path  = model.crf.decode(feats, masks.byte())
            loss = model.loss(feats, masks.byte(), tags)
            eval_loss += loss.item()
            pred.extend([t for t in best_path])
            true.extend([[x for x in t.tolist() if x != 0] for t in tags])
    true = [[id2tag[y] for y in x] for x in true]
    pred = [[id2tag[y] for y in x] for x in pred]
    #entity_types_to_filter = ['<start>', '<eos>','<pad>']
    entity_types_to_filter = ['<s>', '</s>','<pad>']
    true, pred = filter_entity_type(true, pred, entity_types_to_filter)
    f1 = f1_score(true, pred)
    if test:
        accuracy = accuracy_score(true, pred)
        precision = precision_score(true, pred)
        recall = recall_score(true, pred)
        report = classification_report(true, pred,4)
        return accuracy, precision, recall, f1, eval_loss / len(dev_loader), report
    model.train()
    return f1, eval_loss / len(dev_loader)
def filter_entity_type(y_true, y_pred, entity_types):
    y_true_filtered = []
    y_pred_filtered = []
    for true_seq, pred_seq in zip(y_true, y_pred):
        true_seq_filtered = [label if all(not label.endswith(entity_type) for entity_type in entity_types) else 'O' for
                             label in true_seq]
        pred_seq_filtered = [label if all(not label.endswith(entity_type) for entity_type in entity_types) else 'O' for
                             label in pred_seq]
        y_true_filtered.append(true_seq_filtered)
        y_pred_filtered.append(pred_seq_filtered)
    return y_true_filtered, y_pred_filtered

def test(model, test_loader, config, id2tag):
    path = 'result/'
    checkpoint_path = os.path.join(path, 'checkpoint')
    
    # 从checkpoint文件中读取模型名称
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, 'r',encoding='utf-8') as file:
            saved_model_name = file.read().strip()
            model_path = os.path.join(path, saved_model_name)
            if os.path.exists(model_path):
                # 加载保存的最优模型
                model.load_state_dict(torch.load(model_path))
                if config.use_cuda:
                    model.cuda()
                print(f"Loaded model '{saved_model_name}' successfully for testing.")
            else:
                print(f"Model path '{model_path}' does not exist. Testing with current model state.")
    else:
        print("Checkpoint file does not exist. Testing with current model state.")
    #model.eval()
    accuracy, precision, recall, f1, loss, report = dev(model=model, dev_loader=test_loader,
                                                        config=config, id2tag=id2tag, test=True)

    ### 输出数据集
    # print("ERNIE-BiLSTM-CRF on PeopleDaliy dataset is done")
    # print("ERNIE-BiLSTM-CRF on MASA dataset is done")
    # print("ERNIE-BiLSTM-CRF on Boson dataset is done")
    # print("ERNIE-BiLSTM-CRF on Weibo dataset is done")
    # print("ERNIE-BiLSTM-CRF on Resume dataset is done")

    # print("ERNIE-BiGRU-CRF on PeopleDaliy dataset is done")
    # print("ERNIE-BiGRU-CRF on MASA dataset is done")
    # print("ERNIE-BiGRU-CRF on Boson dataset is done")
    # print("ERNIE-BiGRU-CRF on Weibo dataset is done")
    # print("ERNIE-BiGRU-CRF on Resume dataset is done")

    # print("ERNIE3.0-BiGRU-CRF on PeopleDaliy dataset is done")
    # print("ERNIE3.0-BiGRU-CRF on MASA dataset is done")
    # print("ERNIE3.0-BiGRU-CRF on Boson dataset is done")
    # print("ERNIE3.0-BiGRU-CRF on Weibo dataset is done")
    # print("ERNIE3.0-BiGRU-CRF on Resume dataset is done")

    # print("ERNIE3.0-BiLSTM-CRF on PeopleDaliy dataset is done")
    # print("ERNIE3.0-BiLSTM-CRF on MASA dataset is done")
    # print("ERNIE3.0-BiLSTM-CRF on Boson dataset is done")
    # print("ERNIE3.0-BiLSTM-CRF on Weibo dataset is done")
    # print("ERNIE3.0-BiLSTM-CRF on Rusume dataset is done")
    # print("LLM+ERNIE3.0-BiLSTM-CRF on Weibo dataset is done")

    #print("ERNIE3.0-BiLSMT-CRF on CHE_TXT dataset is done")
    #print("ERNIE3.0-BiLSMT-MHATT-CRF on CHE_TXT dataset is done")
    # print("ERNIE3.0-BiGRU-CRF on CHE_TXT dataset is done")

    #print("Bert-BiLSTM-CRF on CHE_TXT dataset is done")
    #print("ERNIE-BiGRU--CRF_1024 on CHE_BMEO dataset is done")
    print("robera-BiGRU--CRF_999 on CHE_BMEO dataset is done")
    
    msg1 = 'Test Loss:{0:5.2}, Test Acc:{1:6.2%}'
    print(msg1.format(loss, accuracy))
    msg2 = 'Test Precision:{}, Test Recall:{}, Test F1_Socre:{}'
    print(msg2.format(precision, recall, f1))
    print('Cllassification Report:')
    print(report)

if __name__ == '__main__':
    

    #set_seed(1950)
   
    start_time = datetime.datetime.now()
    # fire.Fire()
    
   
    train()
    end_time= datetime.datetime.now()
    elapsed = end_time - start_time
    print("start_time:" + start_time.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3])
    print("end_time:" + end_time.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3])
    print(" spend times ：" + str(int(elapsed.total_seconds()*1000)) + "ms")

#python main.py train --use_cuda=True --batch_size=50 PeopleDaily MASA