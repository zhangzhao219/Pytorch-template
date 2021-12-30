import torch
import torch.nn as nn
import torch.nn.functional as F

class BiLSTM_Attention(nn.Module):
    def __init__(self,args,vocab_size,embedding_dim,n_hidden,num_classes):
        super(BiLSTM_Attention, self).__init__()
        self.args = args
        self.n_hidden = n_hidden
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, n_hidden,bidirectional=True)
        self.out = nn.Linear(n_hidden * 2, num_classes)

    # lstm_output : [batch_size, n_step, n_hidden * num_directions(=2)], F matrix
    def attention_net(self, lstm_output, final_state):
        hidden = final_state.view(-1, self.n_hidden * 2, 1)   # hidden : [batch_size, n_hidden * num_directions(=2), 1(=n_layer)]
        attn_weights = torch.bmm(lstm_output, hidden).squeeze(2) # attn_weights : [batch_size, n_step]
        soft_attn_weights = F.softmax(attn_weights, 1)
        # [batch_size, n_hidden * num_directions(=2), n_step] * [batch_size, n_step, 1] = [batch_size, n_hidden * num_directions(=2), 1]
        context = torch.bmm(lstm_output.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2)
        return context, soft_attn_weights
        return context, soft_attn_weights.data.cpu().numpy() # context : [batch_size, n_hidden * num_directions(=2)]

    def forward(self, X):
        batch_size = X.size(0)
        hidden = self.init_hidden(batch_size)
        input = self.embedding(X) # input : [batch_size, len_seq, embedding_dim]
        input = input.permute(1, 0, 2) # input : [len_seq, batch_size, embedding_dim]

        # final_hidden_state, final_cell_state : [num_layers(=1) * num_directions(=2), batch_size, n_hidden]
        output, (final_hidden_state, final_cell_state) = self.lstm(input, hidden)
        output = output.permute(1, 0, 2) # output : [batch_size, len_seq, n_hidden]
        attn_output, attention = self.attention_net(output, final_hidden_state)
        return self.out(attn_output)
        return self.out(attn_output), attention # model : [batch_size, num_classes], attention : [batch_size, n_step]

    def init_hidden(self,batch_size):
        hidden_state = torch.zeros(1*2, batch_size, self.n_hidden)
        cell_state = torch.zeros(1*2, batch_size, self.n_hidden)
        if self.args.gpu:
            hidden_state = hidden_state.cuda()
            cell_state = cell_state.cuda()
        return (hidden_state,cell_state)
        # [num_layers(=1) * num_directions(=2), batch_size, n_hidden]
        # [num_layers(=1) * num_directions(=2), batch_size, n_hidden]