from libs.modules.includes import *
from libs.modules.thirdPart import AdditiveAttention


class Seq2SeqEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.1):
        super(Seq2SeqEncoder, self).__init__()
        self.embedding = nn.Linear(input_size, hidden_size)
        self.gru = nn.GRU(input_size=hidden_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout)

    def forward(self, x):
        # x : torch.Size([360, 1, 2048])
        x = self.embedding(x)
        out, hidden_state = self.gru(x)
        return out, hidden_state


# ToDo(Alex Han) 尚未完成
class Seq2SeqDecoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Seq2SeqDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Linear(in_features=output_size, out_features=hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden):
        out = self.embedding(x).view(1, 1, -1)
        out = F.relu(out)
        out, hidden = self.gru(out, hidden)
        out = self.out(out[0])
        return out, hidden


class Seq2SeqAttentionDecoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.1):
        super(Seq2SeqAttentionDecoder, self).__init__()
        self.embedding = nn.Linear(input_size, hidden_size)
        self.attention = AdditiveAttention(key_size=hidden_size, query_size=hidden_size,
                                           num_hiddens=hidden_size, dropout=dropout)
        self.gru = nn.GRU(input_size=hidden_size * 2, hidden_size=hidden_size,
                          num_layers=num_layers, dropout=dropout)

        self._attention_weights = []

    def init_state(self, enc_outputs, enc_valid_lens=None, *args):
        # outputs的形状为(batch_size，num_steps，num_hiddens).
        # hidden_state的形状为(num_layers，batch_size，num_hiddens)
        outputs, hidden_state = enc_outputs
        return outputs.permute(1, 0, 2), hidden_state, enc_valid_lens

    def forward(self, decoder_inputs, decoder_state):
        # [num_layers, 1, 2048]
        enc_outputs, hidden_state, enc_valid_lens = decoder_state
        # decoder_inputs :  torch.Size([30, 1440]) -> torch.Size([30, 2048])
        decoder_inputs = self.embedding(decoder_inputs)
        outputs, self._attention_weights = [], []
        for decoder_input in decoder_inputs:
            decoder_input = decoder_input.unsqueeze(dim=0)
            decoder_input = decoder_input.unsqueeze(dim=0)
            # query :  torch.Size([1, 1, 2048])
            query = torch.unsqueeze(hidden_state[-1], dim=1)
            # context :  torch.Size([1, 1, 2048])
            context = self.attention(query, enc_outputs, enc_outputs, enc_valid_lens)  # bahdanau attention
            # decoder_input :  torch.Size([1, 1, 2048]) -> torch.Size([1, 1, 4096])
            decoder_input = torch.cat([context, decoder_input], dim=-1)
            # out :  torch.Size([1, 1, 2048])
            out, hidden_state = self.gru(decoder_input.permute(1, 0, 2), hidden_state)
            outputs.append(out)
            with torch.no_grad():
                # self.attention.attention_weights : torch.Tensor
                # self.attention.attention_weights.shape : torch.Size([1, 1, 360])
                self._attention_weights.append(self.attention.attention_weights)

        # 30*[1, 1, 360] -> [30, 1, 360]
        self._attention_weights = torch.cat(self._attention_weights, dim=0).detach().cpu()
        # outputs: list -> torch.Tensor
        # [torch.Size([1, 1, 2048]), torch.Size([1, 1, 2048]), ...] -> torch.Size([30, 1, 2048])
        return torch.cat(outputs, dim=0), self._attention_weights, [enc_outputs, hidden_state, enc_valid_lens]

    @property
    def attention_weights(self):
        return self._attention_weights

    def clean_attention_weights(self):
        self._attention_weights = []
        torch.cuda.empty_cache()


class Seq2SeqAttention(nn.Module):  # 弃用
    def __init__(self, input_size, hidden_size, output_size, num_layers, dropout=0.1):
        super(Seq2SeqAttention, self).__init__()
        self.encoder = Seq2SeqEncoder(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                                      dropout=dropout)
        self.decoder = Seq2SeqAttentionDecoder(input_size=hidden_size, hidden_size=hidden_size, num_layers=num_layers,
                                               dropout=dropout)
        self.dense = nn.Sequential(
            nn.Linear(in_features=hidden_size, out_features=output_size),
            Permute(1, 0, 2)  # 在includes.py中实现，调用 torch.tensor 的 permute 方法实现
        )

    def forward(self, enc_X, dec_X, *args):
        enc_outputs = self.encoder(enc_X, *args)
        # dec_input = torch.cat([enc_outputs[0], dec_X[1:]], dim=)
        dec_state = self.decoder.init_state(enc_outputs, *args)
        outputs, stats = self.decoder(dec_X, dec_state)
        outputs = self.dense(outputs)
        return outputs, stats