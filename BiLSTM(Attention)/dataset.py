import torch
from torch.utils.data import Dataset
class LoadData(Dataset):
    def __init__(self,data,word_dict):
        super(LoadData,self).__init__()
        self.data = data
        self.word_dict = word_dict
        self.len = len(data)
    def __getitem__(self,index):
        data = self.data['sentence'].values[index]
        data = self.padding_sentence(data,10,'<PAD>')
        label = self.data['label'].values[index]
        # UNK_data = []
        # for i in data.split():
        #     if i in self.word_dict:
        #         UNK_data.append(self.word_dict[i])
        #     else:
        #         UNK_data.append(self.word_dict['<UNK>'])
        # return torch.tensor(UNK_data),label
        return torch.tensor([self.word_dict[i] for i in data.split()]),label
    def __len__(self):
        return self.len
    def padding_sentence(self,sentence,max_len,pad_word):
        sentence_list = sentence.split(' ')
        sentence_len = len(sentence_list)
        if sentence_len > max_len:
            sentence_list = sentence_list[:max_len]
            return ' '.join(sentence_list)
        else:
            for i in range(max_len - sentence_len):
                sentence_list.append(pad_word)
            return ' '.join(sentence_list)

