"""Encoder and decoder with attention multifactor version"""


from libs.modules.includes import *
from libs.modules.thirdPart import AdditiveAttention


class Seq2SeqEncodeNoEmbedding(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.1):
        super(Seq2SeqEncodeNoEmbedding, self).__init__()
        # self.embedding = nn.Linear(input_size, hidden_size)
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout)

    def forward(self, x):
        # 输出'X'的形状：(batch_size,num_steps,embed_size)
        # x = self.embedding(x)
        # 在循环神经网络模型中，第一个轴对应于时间步,(batch_size,num_steps,embed_size)->(num_steps,batch_size,embed_size)
        x = x.permute(1, 0, 2)
        out, hidden_state = self.gru(x)
        # output的形状:(num_steps,batch_size,num_hiddens)
        # state[0]的形状:(num_layers,batch_size,num_hiddens)
        return out, hidden_state


class Seq2SeqAttentionDecoderNoEmbedding(nn.Module):
    """我们可以认为，一支股票的各个因子（高开低收价、均线、动量等等）是一个词的词向量，那么我们不需要embedding层，网络输入输出多个参数，实现多部预测"""
    def __init__(self, input_size, hidden_size, output_size, num_layers, dropout=0.1):
        super(Seq2SeqAttentionDecoderNoEmbedding, self).__init__()
        self.attention = AdditiveAttention(key_size=hidden_size, query_size=hidden_size,
                                           num_hiddens=hidden_size, dropout=dropout)
        self.gru = nn.GRU(input_size=input_size + hidden_size, hidden_size=hidden_size,
                          num_layers=num_layers, dropout=dropout)
        self.dense = nn.Linear(in_features=hidden_size, out_features=output_size, bias=False)

    def init_state(self, enc_outputs):
        # outputs的形状为(batch_size，num_steps，num_hiddens).
        # hidden_state的形状为(num_layers，batch_size，num_hiddens)
        outputs, hidden_state = enc_outputs
        return outputs.permute(1, 0, 2), hidden_state

    def forward(self, decoder_input, decoder_state):
        """
        :param decoder_input: (batch_size,num_steps,embed_size)
        :param decoder_state: [enc_outputs, hidden_state]
        :return:
        """

        # enc_outputs的形状为(batch_size,num_steps,num_hiddens).
        # hidden_state的形状为(num_layers,batch_size,num_hiddens)
        enc_outputs, hidden_state = decoder_state

        # decoder_input : (batch_size, steps, hidden_size)
        # decoder_input = self.embedding(decoder_input)

        # decoder_input : (steps, batch_size, hidden_size)
        decoder_input = decoder_input.permute(1, 0, 2)

        # query :  torch.Size([1, 1, hidden_size])
        query = torch.unsqueeze(hidden_state[-1], dim=1)  # 使用上一步的隐藏状态
        # context :  torch.Size([1, 1, hidden_size])
        context = self.attention(queries=query, keys=enc_outputs, values=enc_outputs)  # bahdanau attention

        # print("query : ", query.shape)
        # print("context : ", context.shape)
        # print("decoder_input : ", decoder_input.shape)

        # decoder_input: torch.Size([1, 1, 2 * hidden_size])
        decoder_input = torch.cat([context, decoder_input], dim=-1)

        # out : (num_steps,batch_size,num_hiddens).
        out, hidden_state = self.gru(decoder_input.permute(1, 0, 2), hidden_state)
        out = self.dense(out)

        return out, self.attention.attention_weights, [enc_outputs, hidden_state]



