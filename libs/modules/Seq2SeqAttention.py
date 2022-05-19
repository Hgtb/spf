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
        自动选择运行模式,每次处理一个batch的数据
        :param enc_input: History data
        :param dec_input: The data used in teacher forcing, the length of `teacher_data` is the prediction steps
        :param steps: The prediction steps in evaluate_module
        :return: module output data, shape is torch.Tensor([len(teacher_data) or steps, 1 output_size])
        """
        if self.training & (dec_input is None):
            raise "The mode of the model is training, but the parameter 'teacher_data' is not passed in."
        if (not self.training) & (steps is None):
            raise "The mode of the model is evaluating, but the parameter 'steps' is not passed in."

        if self.training:
            return self.train_module(enc_input, dec_input)
        else:
            return self.evaluate_module(enc_input, steps)

    def train_module(self, enc_inputs: torch.Tensor, dec_inputs: torch.Tensor):
        """
        :param enc_inputs: (batch_size,num_steps,embed_size)
        :param dec_inputs: (batch_size,num_steps,embed_size)
        :return:
        """
        output_seq = []
        attention_weight_seq = []

        # encoder_output的形状:(num_steps,batch_size,num_hiddens)
        encoder_output, encoder_state = self.encoder(enc_inputs)
        decoder_states = self.decoder.init_state(enc_outputs=(encoder_output, encoder_state))

        # enc_inputs : (enc_steps, batch_size, embed_size)
        # dec_inputs : (dec_steps, batch_size, embed_size)
        enc_inputs = enc_inputs.permute(1, 0, 2)
        dec_inputs = dec_inputs.permute(1, 0, 2)

        # 拼接encoder的最后一步输入和decoder的第一至倒数第二步,作为decoder的输入
        # dec_inputs : (dec_steps, batch_size, embed_size)
        dec_inputs = torch.cat([enc_inputs[-1].unsqueeze(dim=0), dec_inputs[:-1]], dim=0)

        for step in range(len(dec_inputs)):
            # dec_input : (batch_size=1, steps=1, input_size)
            dec_input = dec_inputs[step].unsqueeze(dim=0)
            dec_output, decoder_attention_weights, decoder_states = self.decoder(dec_input, decoder_states)
            dec_output = self.dense(dec_output)

            output_seq.append(dec_output)
            attention_weight_seq.append(decoder_attention_weights)

        # output_seq : (dec_steps, batch_size, output_size)
        # attention_weight_seq : (steps, 1, enc_steps)
        output_seq = torch.cat(output_seq, dim=0)
        attention_weight_seq = torch.cat(attention_weight_seq, dim=0).detach().cpu()

        # decoder_states : [enc_outputs, hidden_state]
        return output_seq, attention_weight_seq, decoder_states

    def evaluate_module(self, enc_inputs: torch.Tensor, steps: int):
        """
        :param enc_inputs: (batch_size,num_steps,embed_size)
        :param steps:
        :return:
        """

        output_seq = []
        attention_weight_seq = []

        # output的形状:(num_steps,batch_size,num_hiddens)
        encoder_output, encoder_state = self.encoder(enc_inputs)
        decoder_states = self.decoder.init_state(enc_outputs=(encoder_output, encoder_state))

        # dec_inputs : (dec_steps, batch_size, embed_size)
        enc_inputs = enc_inputs.permute(1, 0, 2)

        # 使用encoder的最后一步作为输入
        dec_input = enc_inputs[-1].unsqueeze(dim=0)

        for step in range(steps):
            dec_output, decoder_attention_weights, decoder_states = self.decoder(dec_input, decoder_states)
            dec_input = self.dense(dec_output)

            output_seq.append(dec_input)
            attention_weight_seq.append(decoder_attention_weights)

        # output_seq : (steps, batch_size, output_size)
        # attention_weight_seq : (steps, ...
        output_seq = torch.cat(output_seq, dim=0)
        attention_weight_seq = torch.cat(attention_weight_seq, dim=0).detach().cpu()

        # decoder_states : [enc_outputs, hidden_state]
        return output_seq, attention_weight_seq, decoder_states


def train_Seq2SeqAttention(model: Seq2SeqAttention,
                           device: torch.device,
                           lr: float,
                           loss_function_name: str,
                           num_epochs: int,
                           dataLoader,
                           init_weight: bool = True,
                           use_scheduler: bool = False
                           ):
    best_loss = 10
    best_model = None
    def init_weights(m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
        if type(m) == nn.GRU:
            for param in m._flat_weights_names:
                if "weight" in param:
                    nn.init.orthogonal_(m._parameters[param])

    model.to(device)
    if init_weight:
        model.apply(init_weights)
    dataLoader.to_device(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode="min", factor=0.5, patience=5,
                                                           cooldown=50)
    loss = None
    try:
        loss = loss_function(loss_function_name)(gamma=0.1)
    finally:
        loss = loss_function(loss_function_name)()
    ls = []  # 记录每一batch的loss
    model.train()

    for epoch in range(num_epochs):
        dataLoader.reset()
        tqdm_ = tqdm(dataLoader, desc=f"epoch {epoch} training")
        for X, Y, _ in tqdm_:
            # X : torch.Size([enc_steps, stock_num, parameters_num])
            # Y : torch.Size([dec_steps, stock_num, parameters_num])
            optimizer.zero_grad()

            enc_steps, _, _ = X.shape
            dec_steps, _, _ = Y.shape

            # input_size = stock_num * parameters_num
            # X : (enc_steps, input_size)
            # Y : (dec_steps, input_size)
            X = X.reshape(enc_steps, -1)
            Y = Y.reshape(dec_steps, -1)

            # batch_size = 1
            # X : torch.Size([batch_size, enc_steps, input_size])
            # Y : torch.Size([batch_size, dec_steps, input_size])
            X = X.unsqueeze(dim=0)
            Y = Y.unsqueeze(dim=0)

            # Y_hat : (dec_steps, batch_size, output_size)
            Y_hat, _, _ = model(enc_input=X, dec_input=Y)

            # Y_hat : (batch_size, dec_steps, input_size)
            Y_hat = Y_hat.permute(1, 0, 2)
            l = loss(Y_hat, Y).sum()
            if abs(l.cpu().item()) < best_loss:
                best_loss = abs(l.cpu().item())
                best_model = model
            l.backward()
            ls.append(l.detach().cpu().item())

            grad_clipping(model, 1)
            optimizer.step()
            if use_scheduler:
                scheduler.step(l.sum())

            # tqdm 进度条更新 loss, lr
            tqdm_.set_postfix(loss=l.detach().item(), lr=optimizer.param_groups[0]['lr'])
    return best_model, ls


def eval_Seq2SeqAttention(model: Seq2SeqAttention,
                          device: torch.device,
                          dataLoader,
                          steps: int,  # dec_steps
                          loss_function_name: str = None):
    torch.no_grad()
    model.eval()
    model.to(device)
    dataLoader.to_device(device)

    predict_seq = []
    target_seq = []
    attention_seq = []

    for model_input, target_data, _ in tqdm(dataLoader):
        # model_input : torch.Size([enc_steps, stock_num, parameters_num])
        # target_data : torch.Size([dec_steps, stock_num, parameters_num])

        enc_steps, stock_num, _ = model_input.shape
        dec_steps, stock_num, _ = target_data.shape

        # input_size = stock_num * parameters_num
        # X : (enc_steps, input_size)
        model_input = model_input.reshape(enc_steps, -1)
        target_data = target_data.reshape(dec_steps, -1)

        # batch_size = 1
        # model_input : torch.Size([batch_size, enc_steps, input_size])
        # model_input : torch.Size([batch_size, dec_steps, output_size])
        model_input = model_input.unsqueeze(dim=0)
        target_data = target_data.unsqueeze(dim=0)

        # output_seq : (steps, batch_size, output_size)
        model_output, attention_weight, _ = model(enc_input=model_input, steps=steps)

        # model_output : (1, dec_steps, output_size)
        # target_data  : (1, dec_steps, output_size)
        # attention_weight : (1, dec_steps, enc_steps)
        model_output = model_output.detach().cpu().permute(1, 0, 2)
        target_data = target_data.cpu()
        attention_weight = attention_weight.detach().cpu().permute(1, 0, 2)

        predict_seq.append(model_output)
        target_seq.append(target_data)
        attention_seq.append(attention_weight)

    # predict_seq   : (dataLoader_len, dec_steps, output_size)
    # target_seq    : (dataLoader_len, dec_steps, output_size)
    # attention_seq : (dataLoader_len, dec_steps, 1, enc_steps)
    predict_seq = torch.cat(predict_seq, dim=0)
    target_seq = torch.cat(target_seq, dim=0)
    attention_seq = torch.cat(attention_seq, dim=0)

    # len(predict_seq) = len(target_seq) = len9attention_seq) = steps
    return predict_seq, target_seq, attention_seq
