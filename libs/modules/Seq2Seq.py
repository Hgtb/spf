from libs.modules.includes import *


class Seq2SeqEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.1):
        super(Seq2SeqEncoder, self).__init__()
        self.embedding = nn.Linear(input_size, hidden_size)
        self.gru = nn.GRU(input_size=hidden_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout)

    def forward(self, x):
        # x : torch.Size([360, 1, 2048])
        x = self.embedding(x)
        out, hidden_state = self.gru(x)
        # print(out.shape)
        # print(hidden_state.shape)
        return out, hidden_state


class Seq2SeqDecoder(nn.Module):
    def __init__(self, hidden_size, output_size):
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


# ToDo(Alex Han) 完成AttentionDecoder
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
        # print("Seq2Seq&Attention Decoder Input : ", decoder_inputs.shape)
        enc_outputs, hidden_state, enc_valid_lens = decoder_state
        decoder_inputs = self.embedding(decoder_inputs)
        outputs, self._attention_weights = [], []
        for decoder_input in decoder_inputs:
            decoder_input = decoder_input.reshape(1, 1, -1)  # [2048] -> [1, 1, 2048]
            query = torch.unsqueeze(hidden_state[-1], dim=1)
            context = self.attention(query, enc_outputs, enc_outputs, enc_valid_lens)  # bahdanau attention
            # print("context : ", context.shape)
            # print("decoder_input : ", decoder_input.shape)
            # print("torch.unsqueeze(decoder_input, dim=1) : ", torch.unsqueeze(decoder_input, dim=1).shape)
            decoder_input = torch.cat([context, decoder_input], dim=-1)
            # print("x : ", decoder_input.shape)
            # print("x.permute(1, 0, 2) : ", decoder_input.permute(1, 0, 2).shape)
            out, hidden_state = self.gru(decoder_input.permute(1, 0, 2), hidden_state)
            # print("out : ", out.shape)
            outputs.append(out)
            with torch.no_grad():
                # self.attention.attention_weights : torch.Tensor
                # self.attention.attention_weights.shape : torch.Size([1, 1, 360])
                self._attention_weights.append(self.attention.attention_weights)
        self._attention_weights = torch.cat(self._attention_weights, dim=0).cpu()  # 30*[1, 1, 360] -> [30, 1, 360]
        # outputs: list -> torch.Tensor
        # [torch.Size([1, 1, 2048]), torch.Size([1, 1, 2048]), ...] -> torch.Size([30, 1, 2048])
        return torch.cat(outputs, dim=0), self._attention_weights, [enc_outputs, hidden_state, enc_valid_lens]

    @property
    def attention_weights(self):
        return self._attention_weights

    def clean_attention_weights(self):
        self._attention_weights = []
        torch.cuda.empty_cache()


class Seq2SeqAttention(nn.Module):
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