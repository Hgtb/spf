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
        self.decoder = Seq2SeqAttentionDecoder(input_size=input_size, hidden_size=hidden_size,
                                               num_layers=num_layers, dropout=dropout)
        self.dense = nn.Linear(in_features=hidden_size, out_features=output_size)


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
        # torch.Size([1, 1, 1440]) + torch.Size([29, 1, 1440]) -> torch.Size([30, 1, 1440])
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

            # decoder_attention_weights : torch.Size([5, 1, 360])
            output_seq.append(decoder_input)
            attention_weight_seq.append(decoder_attention_weights)

        # output_seq : torch.Size([steps, 1, output_size)]
        output_seq = torch.cat(output_seq, dim=0)
        attention_weight_seq = torch.cat(attention_weight_seq, dim=0)

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
                    nn.init.orthogonal_(m._parameters[param])

    if init_weights:
        model.apply(xavier_init_weights)
    model.to(device)
    dataLoader.to_device(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode="min", factor=0.5, patience=5,
                                                           cooldown=100)
    loss = loss_function(loss_function_name)()
    ls = []
    l = torch.Tensor([0])
    model.train()

    for epoch in range(num_epochs):
        dataLoader.resetShifter()
        tqdm_ = tqdm(dataLoader, desc=f"epoch {epoch} training")
        for X, Y, _ in tqdm_:
            # X : torch.Size([input_steps, stock_num, parameters_num])
            # Y : torch.Size([predict_steps, stock_num, parameters_num])
            # input_size == output_size = 1440
            # batch_size = 1
            # input_steps = 360
            # predict_steps = 30
            optimizer.zero_grad()
            X = X.reshape(360, 1, -1)  # (input_steps, 1, stock_num * parameters_num)
            Y = Y.reshape(5, 1, -1)  # (predict_steps, 1, stock_num * parameters_num)

            Y_hat, attention_weight_seq, _ = model(enc_input=X, dec_input=Y)
            l = loss(Y_hat, Y).sum()
            l.backward()  # 有bug
            ls.append(l.detach().cpu().item())

            # print("Loss : ", l.sum())
            grad_clipping(model, 1)
            optimizer.step()
            scheduler.step(l.sum())

            # tqdm 进度条更新 loss
            tqdm_.set_postfix(loss=l.detach().item(), lr=optimizer.param_groups[0]['lr'])
    return model, ls


def eval_Seq2SeqAttention(model: Seq2SeqAttention,
                          device: torch.device,
                          dataLoader,
                          steps: int,  # predict_steps
                          loss_function_name: str = None):
    torch.no_grad()
    model.eval()
    model.to(device)
    dataLoader.to_device(device)
    predict_seq = []
    target_seq = []
    attention_seq = []
    for model_input, target_data, _ in dataLoader:
        # model_input : torch.Size([input_steps, stock_num, parameters_num])
        # target_data : torch.Size([predict_steps, stock_num, parameters_num])
        model_input = model_input.reshape(360, 1, -1)
        model_output, attention_weight, _ = model(enc_input=model_input, steps=steps)
        predict_seq.append(model_output.detach().cpu().permute(2, 0, 1).squeeze())  # (30, 1, 1440) -> (1440, 5)
        # print("model_output.detach().cpu().permute(2, 0, 1).squeeze() : ",
        #       model_output.detach().cpu().permute(2, 0, 1).squeeze().shape)
        target_seq.append(target_data.cpu().permute(1, 0, 2).squeeze())  # (30, 1440) -> (1440, 5)
        # print("target_data.cpu().permute(1, 0, 2).squeeze() : ", target_data.cpu().permute(1, 0, 2).squeeze().shape)
        attention_seq.append(attention_weight.detach().cpu())

    # len(predict_seq) = len(target_seq) = len9attention_seq) = steps
    return predict_seq, target_seq, attention_seq
