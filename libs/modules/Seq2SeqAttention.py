import torch

from libs.modules.includes import *
from libs.modules.Seq2Seq import *
from libs.modules.Loss import *


class Seq2SeqAttention(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, dropout):
        """
        input_size -> hidden_size ---------------------> hidden_size -> output_size
        |-----encoder-----|--------------decoder--------------|-------dense-------|
        """
        super(Seq2SeqAttention, self).__init__()
        self.encoder = Seq2SeqEncoder(input_size=input_size, hidden_size=hidden_size,
                                      num_layers=num_layers, dropout=dropout)
        self.decoder = Seq2SeqAttentionDecoder(input_size=hidden_size, hidden_size=hidden_size,
                                               num_layers=num_layers, dropout=dropout)
        self.dense = nn.Linear(in_features=input_size, out_features=output_size)

    def forward(self, enc_input, dec_input=None, steps=None):
        """
        It can automatically select the operating mode of the decoder
        :param enc_input: History data
        :param dec_input: The data used in teacher forcing, the length of `teacher_data` is the prediction steps
        :param steps: The prediction steps in evaluate_module
        :return: module output data, shape is torch.Tensor([len(teacher_data) or steps, 1 output_size])
        """
        # if x.shape != torch.Size([360, 1, 120, 120]):
        #     raise f"Module input shape error.Expect torch.Size([360, 1, 120, 120]), got {str(x.shape)}."
        if self.training & (dec_input is None):
            raise "The mode of the model is training, but the parameter 'teacher_data' is not passed in."
        if (not self.training) & (steps is None):
            raise "The mode of the model is evaluating, but the parameter 'steps' is not passed in."

        if self.training:
            return self.train_module(enc_input, dec_input)
        else:
            return self.evaluate_module(enc_input, steps)

    def train_module(self, enc_input, dec_input):
        # enc_input : (seq_len, batch_size, input_size)
        # dec_input : (seq_len, batch_size, input_size)
        encoder_output, encoder_state = self.encoder(enc_input)
        decoder_states = self.decoder.init_state(enc_outputs=(encoder_output, encoder_state))
        dec_input = torch.cat([enc_input[-1], dec_input[:-1]], dim=0)  # 使用Encoder的最后一个输入作为开始(向前移动一步)
        decoder_output, decoder_attention_weights, decoder_states = self.decoder(dec_input, decoder_states)
        self.decoder.clean_attention_weights()
        decoder_output = self.dense(decoder_output)

        # decoder_output : torch.Size([steps, batch_size, output_size])
        # decoder_attention_weights :
        # decoder_states : [enc_outputs, hidden_state, enc_valid_lens]
        return decoder_output, decoder_attention_weights, decoder_states

    def evaluate_module(self, enc_input, steps):
        encoder_output, encoder_state = self.encoder(enc_input)
        decoder_state = self.decoder.init_state(enc_outputs=(encoder_output, encoder_state))


        return decoder_output, decoder_attention_weights, decoder_states


def train_Seq2SeqAttention(model: Seq2SeqAttention,
                           device: torch.device,
                           lr: float,
                           loss_function_name: str,
                           num_epochs: int,
                           dataLoader,
                           init_weights: bool = True,
                           ):
    def xavier_init_weights(m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
        if type(m) == nn.GRU:
            for param in m._flat_weights_names:
                if "weight" in param:
                    nn.init.xavier_uniform_(m._parameters[param])
    model.apply(xavier_init_weights)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss = loss_function(loss_function_name)
    model.train()

    for epoch in range(num_epochs):

        for X, Y in dataLoader:
            optimizer.zero_grad()
            X = X.to(device)
            Y = Y.to(device)


    return model


def eval_Seq2SeqAttention(model: Seq2SeqAttention):
    return
