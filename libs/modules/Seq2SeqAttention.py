import torch

from libs.modules.includes import *
from libs.modules.Seq2Seq import *
from libs.modules.Loss import *
from libs.modules.thirdPart import grad_clipping


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

    def forward(self, enc_input: torch.Tensor, dec_input: torch.Tensor = None, steps: int = None):
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

    def train_module(self, enc_input: torch.Tensor, dec_input: torch.Tensor):
        # enc_input : (seq_len, batch_size, input_size)
        # dec_input : (seq_len, batch_size, input_size)
        encoder_output, encoder_state = self.encoder(enc_input)
        decoder_states = self.decoder.init_state(enc_outputs=(encoder_output, encoder_state))
        print("enc_input[-1].unsqueeze(dim=0) : ", enc_input[-1].unsqueeze(dim=0).shape)
        print("dec_input[:-1]] : ", dec_input[:-1].shape)
        dec_input = torch.cat([enc_input[-1].unsqueeze(dim=0), dec_input[:-1]], dim=0)  # 使用Encoder的最后一个输入作为开始(向前移动一步)
        decoder_output, decoder_attention_weights, decoder_states = self.decoder(dec_input, decoder_states)
        self.decoder.clean_attention_weights()
        decoder_output = self.dense(decoder_output)

        # decoder_output : torch.Size([steps or len(dec_input), batch_size, output_size])
        # decoder_attention_weights :
        # decoder_states : [enc_outputs, hidden_state, enc_valid_lens]
        return decoder_output, decoder_attention_weights, decoder_states

    def evaluate_module(self, enc_input: torch.Tensor, steps: int):
        encoder_output, encoder_state = self.encoder(enc_input)
        decoder_states = self.decoder.init_state(enc_outputs=(encoder_output, encoder_state))
        decoder_input = enc_input[-1].unsqueeze(dim=0)  # decoder_input : torch.Size([1, 1, input_size])
        output_seq, attention_weight_seq = [], []
        for step in range(steps):
            # decoder_output : torch.Size([1, 1, hidden_size])

            decoder_output, decoder_attention_weights, decoder_states = self.decoder(decoder_input, decoder_states)
            decoder_input = self.dense(decoder_output)  # decoder_input : torch.Size([1, 1, output_size])

            output_seq.append(decoder_input)
            attention_weight_seq.append(decoder_attention_weights)

        # output_seq : torch.Size([steps, 1, output_size)]
        output_seq = torch.cat(output_seq, dim=0)
        attention_weight_seq = torch.cat(attention_weight_seq, dim=0)
        print("attention_weight_seq : ", attention_weight_seq.shape)

        return output_seq, attention_weight_seq, decoder_states


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
    loss = loss_function(loss_function_name)()
    model.train()

    for epoch in range(num_epochs):
        for X, Y in tqdm(dataLoader, desc=f"epoch {epoch} training"):
            # X : torch.Size([input_steps, batch_size, input_size])
            # Y : torch.Size([predict_steps, output_size])
            # input_size == output_size = 1440
            # batch_size = 1
            # input_steps = 360
            # predict_steps = 30
            optimizer.zero_grad()
            X = X.reshape(360, 1, -1)
            X = X.to(device)
            Y = Y.to(device)

            Y_hat, attention_weight_seq, _ = model(enc_input=X, dec_input=Y)
            l = loss(Y_hat, Y.unsqueeze(dim=1))  # Y : torch.Size([30, 1440]) -> torch.Size([30, 1, 1440])
            l.sum().backward()  # 有bug
            grad_clipping(model, 1)
            optimizer.step()
    return model


def eval_Seq2SeqAttention(model: Seq2SeqAttention,
                          device: torch.device,
                          dataLoader,
                          steps: int,
                          loss_function_name: str = None):
    torch.no_grad()
    model.eval()
    predict_seq = []
    target_seq = []
    attention_seq = []
    for model_input, target_data in dataLoader:
        model_input.to(device)

        model_output, attention_weight, _ = model(enc_input=model_input, steps=steps)

        predict_seq.append(model_output.detach().cpu().permute(2, 0, 1).squeeze())  # (30, 1, 1440) -> (1440, 30)
        target_seq.append(target_data.permute(1, 0))  # (30, 1440) -> (1440, 30)
        attention_seq.append(attention_weight.detach().cpu())

    return predict_seq, target_seq, attention_seq
