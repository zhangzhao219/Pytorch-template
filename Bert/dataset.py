import torch
from torch.utils.data import Dataset
class LoadData(Dataset):
    def __init__(self,data,tokenizer,max_len=180):
        super(LoadData,self).__init__()
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

        self.len = len(self.data)
    def __getitem__(self,index):
        text = self.data['sentence'].values[index]
        input_ids = self.tokenizer(
            text,
            add_special_tokens = True,  # 添加special tokens， 也就是CLS和SEP
            max_length = self.max_len,           # 设定最大文本长度
            padding='max_length',   # pad到最大的长度  
            return_tensors = 'pt'       # 返回的类型为pytorch tensor
        )
        input_ids = {k:v.squeeze(0) for k,v in input_ids.items()}
        # input_ids['input_ids'] = input_ids['input_ids'].squeeze(0)
        label = self.data['label'].values[index]
        # attention_mask=(input_ids > 0)
        # token_type_ids=torch.tensor([0] * len(input_ids))
        return input_ids,label

    def __len__(self):
        return self.len
    def len(self):
        return self.len