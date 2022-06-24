import os
import tqdm
import torch
import random
import logging
import argparse
import numpy as np
import pandas as pd

import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from sklearn.model_selection import KFold
from tensorboardX import SummaryWriter
from transformers import AdamW, get_linear_schedule_with_warmup

from Bert_Model import Bert_Model,get_tokenizer
from dataset import LoadData
from log import config_logging

parser = argparse.ArgumentParser(description='Learning Pytorch NLP')
parser.add_argument('--batch', type=int, default=16, help='Define the batch size')
parser.add_argument('--board', action='store_true', help='Whether to use tensorboard')
parser.add_argument('--checkpoint',type=int, default=0, help='Use checkpoint')
parser.add_argument('--data_dir',type=str, required=True, help='Data Location')
parser.add_argument('--epoch', type=int, default=5, help='Training epochs')
parser.add_argument('--gpu', type=str, nargs='+', help='Use GPU')
parser.add_argument('--K', type=int, default=1, help='K-fold')
parser.add_argument('--load', action='store_true', help='load from checkpoint')
parser.add_argument('--load_pt', type=str, help='load from checkpoint')
parser.add_argument('--lr',type=float, default=0.001, help='learning rate')
parser.add_argument('--predict', action='store_true', help='Whether to predict')
parser.add_argument('--save', action='store_true', help='Whether to save model')
parser.add_argument('--seed',type=int, default=42, help='Random Seed')
parser.add_argument('--test', action='store_true', help='Whether to test')
parser.add_argument('--train', action='store_true', help='Whether to train')
parser.add_argument('--warmup',type=float, default=0.1, help='warm up ratio')
args = parser.parse_args()

# log
config_logging("log")
logging.info('Log is ready!')

if args.board:
    writer = SummaryWriter()
    logging.info('Tensorboard is ready!')

if args.gpu:
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(args.gpu)
    logging.info('GPU:' + ','.join(args.gpu))

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if args.gpu:
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    logging.info('Seed:' + str(seed))

# () read data and process
def read_data(data_path):
    data = pd.read_csv(data_path,names=['id','sentence','label'])
    logging.info('Read data: ' + data_path)
    return data

# # () build vocabulary
# def build_vocab(data):
#     word_list = []
#     # find all words
#     for index,column in data.iterrows():
#         for word in column['sentence'].split():
#             word_list.append(word)
#     # remove duplicate words
#     word_list = sorted(list(set(word_list)))
#     # {word : index}
#     word_dict = {word:index+1 for index,word in enumerate(word_list)}
#     word_dict['<PAD>'] = 0
#     word_dict['<UNK>'] = len(word_dict)
#     id_to_word_dict = {i:j for j,i in word_dict.items()}
#     logging.info('Build Vocab')
#     return word_dict,id_to_word_dict

def train_one_epoch(args,train_loader,model,optimizer,scheduler,criterion,epoch,K):
    # set train mode
    model.train()
    # zero grad
    optimizer.zero_grad()
    loss = 0

    # define tqdm
    epoch_iterator = tqdm.tqdm(train_loader, desc="Iteration", total=len(train_loader))
    # set description of tqdm
    epoch_iterator.set_description(f'Train-{epoch}-{K}')

    for step, (input_ids,label) in enumerate(epoch_iterator):
        model.zero_grad()
        # print(sentence,label)
        if args.gpu:
            input_ids = {k:v.cuda() for k,v in input_ids.items()}
            label = label.cuda()
        if args.board and step == 0 and epoch == 1 and (K == 1 or K == 0):
            writer.add_graph(bilstm_attention,(input_ids,label))
        # print(input_ids)
        # print(input_ids.size())
        # print((input_ids>0).size())
        output = model(input_ids=input_ids,label=label)
        # print(output)
        loss_single = output.loss.cpu()
        # if args.gpu:
        #     output = output.cpu()
        # loss_single = criterion(output, label)
        # print(loss_single)
        loss += loss_single.item()

        # renew tqdm
        epoch_iterator.update(1)
        # add description in the end
        epoch_iterator.set_postfix(loss=loss_single.item())

        # backward 
        loss_single.backward()
        optimizer.step()
        scheduler.step()

    return loss / args.batch

def eval_one_epoch(args,eval_loader,model,epoch,K):
    # test
    model.eval()
    epoch_iterator = tqdm.tqdm(eval_loader, desc="Iteration", total=len(eval_loader))
    # set description of tqdm
    epoch_iterator.set_description(f'Eval-{epoch}-{K}')

    correct = 0
    all = 0
    for step, (input_ids,label) in enumerate(epoch_iterator):

        if args.gpu:
            input_ids = {k:v.cuda() for k,v in input_ids.items()}
            label = label.cuda()
        with torch.no_grad():
            output = model(input_ids=input_ids,label=label)
            logits = output.logits.cpu()
            label = label.cpu()
        # if args.gpu:
        #     output = output.cpu()

        predict = logits.max(1)[1]
        # print(predict)
        # print(label)
        correct += (predict == label).sum().item()
        all += predict.size()[0]
        # renew tqdm
        epoch_iterator.update(1)
    return correct / all

# train
def train(args,data,model,tokenizer):
    logging.info('Start Training!')
    all_dataset = LoadData(data,get_tokenizer(BERT))
    # loss function
    criterion = nn.CrossEntropyLoss()
    # no_decay = ['bias', 'LayerNorm.weight']
    # optimizer_grouped_parameters = [
    #     {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    #     {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    # ]
    # optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr)
    optimizer = AdamW(model.parameters(), lr=args.lr)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = round(all_dataset.len / args.batch) * args.epoch, num_training_steps = all_dataset.len * args.epoch)
    if args.load:
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # epoch = checkpoint['epoch']
        # K_loss = checkpoint['loss']
    max_accuracy = 0
    for epoch in range(args.epoch):
        K_loss = 0
        K_accuracy = 0
        if args.K >= 2:
            kf = KFold(n_splits=args.K,shuffle=True,random_state=args.seed)
            k = 1
            for train_index, eval_index in kf.split(all_dataset):
                train_dataset = Subset(all_dataset, train_index)
                eval_dataset = Subset(all_dataset, eval_index)
                train_loader = DataLoader(train_dataset,batch_size=args.batch,shuffle=True,drop_last=False)
                eval_loader = DataLoader(eval_dataset,batch_size=args.batch,shuffle=False,drop_last=False)

                K_loss += train_one_epoch(args,train_loader,model,optimizer,scheduler,criterion,epoch+1,k)
                temp_accuracy = eval_one_epoch(args,train_loader,model,epoch+1,k)
                logging.info(f'Train Epoch = {epoch+1} K = {k} Accuracy:{temp_accuracy:{.6}} Loss:{K_loss/k:{.6}}') 

                K_accuracy += eval_one_epoch(args,eval_loader,model,epoch+1,k)
                logging.info(f'Eval Epoch = {epoch+1} K = {k} Accuracy:{K_accuracy/k:{.6}}')        

                k += 1

            K_loss /= args.K
            K_accuracy /= args.K
        else:
            train_loader = DataLoader(all_dataset,batch_size=args.batch,shuffle=True,drop_last=False)
            eval_loader = train_loader
            K_loss = train_one_epoch(args,train_loader,model,optimizer,scheduler,criterion,epoch+1,K=0)
            logging.info(f'Train Epoch = {epoch+1} Loss:{K_loss:{.6}}')

            K_accuracy = eval_one_epoch(args,eval_loader,model,epoch+1,K=0)
            logging.info(f'Eval Epoch = {epoch+1} Accuracy:{K_accuracy:{.6}}')

            K_accuracy = test(args,test_data,model,1,1)
            logging.info(f'Test Epoch = {epoch+1} Accuracy:{K_accuracy:{.6}}')

        if args.checkpoint != 0 and epoch % args.checkpoint == 0:
            torch.save({'epoch': epoch,'model_state_dict': model.state_dict(),'optimizer_state_dict': optimizer.state_dict(),'loss': K_loss}, MODEL_PATH + 'checkpoint_{}_epoch.pt'.format(epoch))
            logging.info(f'checkpoint_{epoch}_epoch.pt Saved!')

        if args.save and K_accuracy > max_accuracy:
            torch.save(model.state_dict(),'best.pt')
            logging.info(f'Test accuracy:{K_accuracy:{.6}} > max_accuracy! Saved!')
            max_accuracy = K_accuracy

        if args.board:
            writer.add_scalar('loss', K_loss, epoch)
            writer.add_scalar('accuracy', K_accuracy, epoch)
# test
def test(args,data,model,tokenizer,sign):
    # set eval mode
    model.eval()
    if sign == 0:
        model.load_state_dict(torch.load("best.pt"))
        logging.info('Load Best Model!')
    test_dataset = LoadData(data,get_tokenizer(BERT))
    test_loader = DataLoader(test_dataset,batch_size=args.batch,shuffle=False,drop_last=False)

    epoch_iterator = tqdm.tqdm(test_loader, desc="Iteration", total=len(test_loader))
    # set description of tqdm
    epoch_iterator.set_description('Test')

    correct = 0
    for step, (input_ids,label) in enumerate(epoch_iterator):

        if args.gpu:
            input_ids = {k:v.cuda() for k,v in input_ids.items()}
            label = label.cuda()
        with torch.no_grad():
            output = model(input_ids=input_ids,label=label)
            logits = output.logits.cpu()
            label = label.cpu()
        # if args.gpu:
        #     output = output.cpu()

        predict = logits.max(1)[1]
        # print(predict)
        # print(label)
        correct += (predict == label).sum().item()

        # renew tqdm
        epoch_iterator.update(1)
    accuracy = correct / test_dataset.__len__()
    return accuracy

# test
def predict(args,data,model,tokenizer):
    # set eval mode
    model.eval()
    model.load_state_dict(torch.load("best.pt"))
    logging.info('Load Best Model!')
    test_dataset = LoadData(data,get_tokenizer(BERT))
    test_loader = DataLoader(test_dataset,batch_size=args.batch,shuffle=False,drop_last=False)

    epoch_iterator = tqdm.tqdm(test_loader, desc="Iteration", total=len(test_loader))
    # set description of tqdm
    epoch_iterator.set_description('Predict')

    result_list = []

    for step, (input_ids,label) in enumerate(epoch_iterator):

        if args.gpu:
            input_ids = {k:v.cuda() for k,v in input_ids.items()}
            label = label.cuda()
        with torch.no_grad():
            output = model(input_ids=input_ids)
            logits = output.logits.cpu()
        # if args.gpu:
        #     output = output.cpu()

        predict = logits.max(1)[1]
        result_list.append(predict)

        # renew tqdm
        epoch_iterator.update(1)
    result = torch.cat([i for i in result_list]).numpy()
    data['target'] = result
    data[['id','target']].to_csv(DATA_PATH + args.data_dir +'/result.csv',index=None)

MODEL_PATH = './models/'
DATA_PATH = './data/'
TRAIN_DATA = DATA_PATH + args.data_dir + '/train_data.csv'
TEST_DATA = DATA_PATH + args.data_dir + '/test_data.csv'

if __name__ == '__main__':
    # set seed
    set_seed(args.seed)

    # print(args.batch, args.epoch, args.eval, args.gpu, args.seed, args.train)

    # read data
    train_data = read_data(TRAIN_DATA)
    test_data = read_data(TEST_DATA)
    
    # build vocab
    # word_dict,id_to_word = build_vocab(pd.concat([train_data, test_data], axis=0))
    # word_dict = build_vocab(train_data)
    # vocab_size = len(word_dict)
    # print(word_dict)

    # config
    # n_step = 3 # number of cells(= number of Step)
    embedding_dim = 100 # embedding size
    n_hidden = 50  # number of hidden units in one cell
    num_classes = 2  # 0 or 1
    BERT = 'bert'

    # bilstm_attention = Bert_Model(num_classes,BERT)
    bilstm_attention = Bert_Model(n_labels=num_classes)
    if args.load:
        checkpoint = torch.load(MODEL_PATH + args.load_pt)
        logging.info(f'Load {args.load_pt}')
    if args.gpu:
        bilstm_attention = bilstm_attention.cuda()
        if len(args.gpu) >= 2:
            bilstm_attention= nn.DataParallel(bilstm_attention)
    if args.train:
        train(args,train_data,bilstm_attention,1)
    if args.test:
        accuracy = test(args,test_data,bilstm_attention,1,0)
        logging.info(f'Test Accuracy: {accuracy:{.6}}')
    if args.predict:
        predict(args,test_data,bilstm_attention,1)
        logging.info('Predict Finished!')