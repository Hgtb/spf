from libs.modules.includes import *
from libs.modules.thirdPart import AdditiveAttention, dilate_loss, MyTimer
from libs.modules.soft_dtw_cuda import SoftDTW
from libs.dataLoader import DataLoader
from libs.modules.ResNet import *
from libs.modules.Seq2Seq import *


class ResNetSeq2SeqAttention(nn.Module):
    def __init__(self,
                 resNet_layer_nums: list,
                 seq2seq_hidden_size: int,
                 seq2seq_num_layer: int,  # The num_layer of the GRU in the encoder and decoder
                 output_size: int,  # word_vector_size or prediction_parameter_num
                 use_checkpoint=False):
        super(ResNetSeq2SeqAttention, self).__init__()
        # ResNet half precision(float16) to reduce GPU memory usage, ResNet's output should convert to float32.
        self.ResNet = ResNet(layer_nums=resNet_layer_nums, use_checkpoint=use_checkpoint)
        # self.ResNet = ResNet(layer_nums=resNet_layer_nums, use_checkpoint=use_checkpoint).half()
        self.encoder = Seq2SeqEncoder(input_size=2048, hidden_size=seq2seq_hidden_size,
                                      num_layers=seq2seq_num_layer, dropout=0.1)
        self.decoder = Seq2SeqAttentionDecoder(input_size=output_size, hidden_size=seq2seq_hidden_size,
                                               num_layers=seq2seq_num_layer, dropout=0.1)

        # The encoder and the decoder share the dense for decoding([*, 2048] -> [*, 1440]).
        self.dense = nn.Linear(in_features=seq2seq_hidden_size, out_features=output_size)

    def forward(self, x: torch.Tensor, teacher_data: torch.Tensor = None, steps: int = None):  # Train Module
        """
        It can automatically select the operating mode of the decoder
        :param x: History data
        :param teacher_data: The data used in teacher forcing, the length of `teacher_data` is the prediction steps
        :param steps: The prediction steps in evaluate_module
        :return: module output data, shape is torch.Tensor([len(teacher_data) or steps, 1 output_size])
        """
        if x.shape != torch.Size([360, 1, 120, 120]):
            raise f"Module input shape error.Expect torch.Size([360, 1, 120, 120]), got {str(x.shape)}."
        if self.training & (teacher_data is None):
            raise "The mode of the model is training, but the parameter 'teacher_data' is not passed in."
        if (not self.training) & (steps is None):
            raise "The mode of the model is evaluating, but the parameter 'steps' is not passed in."

        if self.training:
            return self.train_module(x, teacher_data)
        else:
            return self.evaluate_module(x, steps)

    def train_module(self, x: torch.Tensor, target_data: torch.Tensor):
        """Use teacher forcing mechanism"""
        assert x.shape == torch.Size([360, 1, 120, 120])
        # x.requires_grad = True
        # target_data.requires_grad = True

        # res_output.shape : torch.Size([360, 2048, 1, 1])
        res_output = self.ResNet(x)
        del x

        # res_output.shape : torch.Size([360, 1, 2048])
        res_output = res_output.reshape(360, 1, 2048)
        # res_output = res_output.reshape(360, 1, 2048).float()
        # print("res_output : ", res_output)

        # enc_output.shape : torch.Size([360, 1, 2048])
        enc_output, enc_state = self.encoder(res_output)
        # print("enc_output : ", enc_output[0])
        del res_output

        # dec_inputs.shape : torch.Size([30, 1440])
        dec_inputs = torch.cat([self.dense(enc_output[-1]), target_data[1:]], dim=0)
        # print("dec_inputs : ", dec_inputs)

        dec_state = self.decoder.init_state(enc_outputs=(enc_output, enc_state))

        # dec_output.shape : torch.Size([30, 1, 2048])
        dec_output, _, dec_state = self.decoder(decoder_inputs=dec_inputs, decoder_state=dec_state)
        # print("dec_output : ", dec_output[0])
        self.decoder.clean_attention_weights()

        # model_output.shape : torch.Size([30, 1, 1440])
        model_output = self.dense(dec_output)
        # print("model_output : ", model_output[0])
        del dec_output
        # print("train_module DONE")
        return model_output, dec_state

    def evaluate_module(self, x: torch.Tensor, steps: int):
        assert x.shape == torch.Size([360, 1, 120, 120])
        attention_weights = []  # Store attention_weights in every step
        x.requires_grad = False

        # res_output.shape : torch.Size([360, 2048, 1, 1])
        res_output = self.ResNet(x)
        del x

        # res_output.shape : torch.Size([360, 1, 2048])     dtype : float16 -> float32
        res_output = res_output.reshape(360, 1, 2048).float()

        # enc_output.shape : torch.Size([360, 1, 2048])
        enc_output, enc_state = self.encoder(res_output)
        del res_output

        # dec_input.shape : torch.Size([1, 1, 2048])
        dec_input = self.dense(enc_output[-1])
        dec_outputs = []
        dec_state = self.decoder.init_state(enc_outputs=(enc_output, enc_state))

        # print("dec_input : ", dec_input.shape)

        for step in range(steps):
            dec_output, attention_weight, dec_state = self.decoder(dec_input, dec_state)
            # print("dec_output", dec_output.shape)
            dec_input = self.dense(dec_output)
            dec_outputs.append(dec_input)
            # print("dec_outputs", len(dec_outputs))

            attention_weights.append(self.decoder.attention_weights)
            self.decoder.clean_attention_weights()

        model_output = torch.cat(dec_outputs, dim=0)
        # print("model_output: ", dec_outputs.shape)

        # model_output.shape : torch.Size([30, 1, 1440])
        del dec_outputs
        torch.cuda.empty_cache()

        # print("self.decoder.attention_weights[0] : ", self.decoder.attention_weights[0].shape)
        # model_output : tensor.Size([30, 1, 1440])
        # attention_weights : list[list[list[torch.Size([])]]]
        return model_output, attention_weights, dec_state


def train_ResNetSeq2SeqAttention(model: ResNetSeq2SeqAttention,
                                 dataLoader: DataLoader,
                                 init_weights: bool,
                                 loss_function,
                                 learning_rate: float,
                                 num_epoch: int,
                                 pre_days: int,
                                 device: torch.device):
    loss_func_list = ["MSE", "SoftDTW", "L1", "SmoothL1Loss"]
    if loss_function not in loss_func_list:
        raise f"loss_function error"

    def xavier_init_weights(m):
        # ToDo(Alex Han) 添加 BatchNormal2d 和 Conv2d 的初始化
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
        # if type(m) == nn.Conv2d:
        #     nn.init.xavier_uniform_(m.weight, )
        if type(m) == nn.GRU:
            for param in m._flat_weights_names:
                if "weight" in param:
                    nn.init.xavier_uniform_(m._parameters[param])

    # loss_file = open("../StockPriceForecast/loss.csv", "a")
    # loss_file.write("loss\r\n")

    model.train()
    torch.cuda.empty_cache()
    model.to(device)

    if init_weights:
        model.apply(xavier_init_weights)

    dataLoader.to_device(device)

    attention_weight_seq = []

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    loss_func = nn.MSELoss()
    if loss_function == "MSE":
        loss_func = nn.MSELoss()
    elif loss_function == "SoftDTW":
        loss_func = SoftDTW(use_cuda=True, gamma=1.0, normalize=True)
    elif loss_function == "L1":
        loss_func = nn.L1Loss()
    elif loss_function == "SmoothL1Loss":
        loss_func = nn.SmoothL1Loss()
    print("Loss function : ", loss_func)

    Loss = []
    for epoch in range(num_epoch):
        torch.cuda.empty_cache()
        timer = MyTimer()
        # draw loss
        for train_data, target_data in tqdm(dataLoader):
            torch.cuda.empty_cache()
            train_data = train_data.reshape(360, 1, 120, 120)
            train_data.requires_grad = True
            target_data.requires_grad = True
            train_data = train_data
            target_data = target_data
            optimizer.zero_grad()

            # model_output : torch.Size([30, 1, 1440])
            model_output, _ = model(train_data, target_data)

            model_output = model_output.permute(2, 0, 1)  # [30, 1, 1440] -> [1440, 30, 1]
            target_data = torch.unsqueeze(target_data, dim=-1)  # [30, 1440] -> [30, 1440, 1]
            target_data = target_data.permute(1, 0, 2)  # [30, 1440, 1] -> [1440, 30, 1]

            # loss, loss_shape, loss_temporal = loss_func(outputs=model_output, targets=target_data, alpha=alpha,
            #                                             gamma=gamma, device=device)

            # loss : [1440]
            loss = loss_func(model_output, target_data)

            if loss_function == "SoftDTW":
                loss = loss.sum() / len(loss)

            loss.backward()
            optimizer.step()

            Loss.append(loss.cpu().detach())

            print("loss : ", loss.cpu())

            del loss
            torch.cuda.empty_cache()

        print(f"latest loss {Loss[-1]}, cost {timer.stop()} sec on {str(device)}")
    return model, attention_weight_seq, Loss


def predict_ResNetSeq2Seq(model: ResNetSeq2SeqAttention,
                          dataLoader: DataLoader,
                          steps: int,
                          Device: torch.device,
                          alpha: int,
                          gamma: int,
                          save_attention_weights=False):
    model.eval()
    torch.no_grad()
    predict_seq = []  # len(all_predict_seq) = len(dataLoader)
    attention_weight_seq = []  # len(all_attention_weight_seq) = len(dataLoader)

    model.to(Device)
    dataLoader.to_device(Device)

    # Calculate loss to evaluate model performance
    loss_func = dilate_loss
    Loss = []
    Loss_shape = []
    Loss_temporal = []

    for model_inputs, target_data in tqdm(dataLoader):
        model_inputs = model_inputs.reshape(360, 1, 120, 120)
        model_output, attention_weight, _ = model(model_inputs)
        del model_inputs

        model_output = model_output.permute(2, 0, 1)  # [30, 1, 1440] -> [1440, 30, 1]
        target_data = target_data.unsequeeze(dim=-1)  # [30, 1440] -> [30, 1440, 1]
        target_data = target_data.permute(1, 0, 2)  # [30, 1440, 1] -> [1440, 30, 1]

        # loss, loss_shape, loss_temporal = loss_func(outputs=model_output, targets=target_data, alpha=alpha,
        #                                             gamma=gamma, device=Device)
        loss = loss_func(model_output, target_data)
        Loss.append(loss.sum().cpu())
        # Loss_shape.append(loss_shape.sum().cpu())
        # Loss_temporal.append(loss_temporal.sum().cpu())
        # del loss, loss_shape, loss_temporal
        del loss
        torch.cuda.empty_cache()

        model_output = torch.squeeze(model_output)  # [1440, 30, 1] -> [1440, 30]
        predict_seq.append(model_output)
        attention_weight_seq.append(attention_weight)
        break
    return predict_seq, attention_weight_seq, (Loss, Loss_shape, Loss_temporal)


if __name__ == "__main__":
    import torch

    # module = ResNetSeq2Seq(use_checkpoint=True).half()
    # module.cuda()
    #
    # # torch.autograd.set_detect_anomaly(True)
    # torch.cuda.empty_cache()  # 清除无用数据，降低显存占用
    # torch.set_grad_enabled(True)  # 设置是否计算梯度
    #
    # testData = torch.rand([360, 1, 120, 120], requires_grad=True).half().cuda()
    # output = module(testData)
    #
    # torch.cuda.empty_cache()

    # ResNet:
    #   使用数据集 > 6GB
    #   使用相同大小随机张量 > 6GB
    #   关闭梯度 2.1GB
