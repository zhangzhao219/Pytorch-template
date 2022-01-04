import torch
import torch.nn as nn

from transformers import BertTokenizer, BertConfig, BertModel,BertForSequenceClassification
from transformers import XLNetTokenizer, XLNetModel, XLNetConfig
from transformers import AutoModel, AutoConfig, AutoTokenizer

def get_bert(bert_name):
    if bert_name == 'xlnet':
        print('load xlnet-base-cased')
        model_config = XLNetConfig.from_pretrained('xlnet-base-cased')
        model_config.output_hidden_states = True
        bert = XLNetModel.from_pretrained('xlnet-base-cased', config=model_config)
    elif bert_name == 'ernie':
        print('load ernie-base-uncased')
        model_config = AutoConfig.from_pretrained('nghuyong/ernie-1.0')
        model_config.output_hidden_states = True
        bert = AutoModel.from_pretrained('nghuyong/ernie-1.0', config=model_config)
    else:
        print('load bert-base-uncased')
        # model_config = BertConfig.from_pretrained('bert-base-uncased')
        # model_config.output_hidden_states = True
        bert = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2, output_attentions=False, output_hidden_states=False)
    return bert

def get_tokenizer(bert_name):
    if bert_name == 'xlnet':
        print('load xlnet-base-cased tokenizer')
        tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')
    elif bert_name == 'ernie':
        print('load ernie-base-cased tokenizer')
        tokenizer = AutoTokenizer.from_pretrained('nghuyong/ernie-1.0')
    else:
        print('load bert-base-uncased tokenizer')
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased',do_lower_case=True)
    return tokenizer

class Bert_Model(nn.Module):
    def __init__(self, n_labels, bert='bert-base', feature_layers=5, dropout=0.5):
        super(Bert_Model, self).__init__()
        self.bert_name = bert
        self.bert = get_bert(bert)
        self.feature_layers = feature_layers
        # self.drop_out = nn.Dropout(dropout)

        # self.l0 = nn.Linear(self.feature_layers*self.bert.config.hidden_size, n_labels)

    def forward(self,input_ids,label=None):
        outs = self.bert(**input_ids,labels=label)
        # out = torch.cat([outs[-i][:, 0] for i in range(1, self.feature_layers+1)], dim=-1)
        # return self.l0(self.drop_out(out))
        return outs