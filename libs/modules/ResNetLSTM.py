from libs.modules.includes import *
from libs.modules.thirdPart import AdditiveAttention, dilate_loss, MyTimer
from libs.dataLoader import DataLoader

"""
ResNet use ResNet50
"""


class BasicBlock(nn.Module):
    """
    ResNet 18 or 34  block
    -> Copy -+-> Conv -> ReLU -> Conv -+-> Plus ->
             |_________________________|
    """

    def __init__(self, in_channels, out_channels, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1),
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), stride=(1, 1),
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.ReLU = nn.ReLU()
        self.downsample = downsample

    def forward(self, x: torch.Tensor):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.ReLU(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.ReLU(out)
        return out


class BottleneckBlock(nn.Module):
    """
    ResNet 50 or 101 or 152  block
    --+-> Conv1*1 -> BN ->  ReLU -> Conv3*3 -> BN -> ReLU -> Conv1*1 -> BN -+-> ReLU ->
      +--------------------------(Conv1*1 -> BN)----------------------------+
                                  自动选择是否卷积x
    """

    def __init__(self, in_channels, hidden_channels, out_channels, stride=1, downsample=None):
        """
        当 in_channels != out_channels 时，自动对x进行卷积升维。
        当 stride > 1 时，若没有传入 downsample ，则使用 conv1*1 stride=2 来对x进行下采样，统一尺寸
        """
        super(BottleneckBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=hidden_channels, kernel_size=(1, 1), stride=(1, 1),
                      padding=0, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=(3, 3),
                      stride=(stride, stride), padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU()
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_channels, out_channels=out_channels, kernel_size=(1, 1), stride=(1, 1),
                      padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

        # ToDo(Alex Han) 在自动网络生成功能完成时，需要将这里的 downsample 模块放到自动网络生成模块中，并从 downSample 传入
        self.downsample = downsample
        if in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 1),
                          stride=(stride, stride), padding=0, bias=False),
                nn.BatchNorm2d(out_channels)
            )

        self.ReLU = nn.ReLU()

    def forward(self, x: torch.Tensor):
        residual = x
        out = self.conv3(self.conv2(self.conv1(x)))

        if self.downsample is not None:
            residual = self.downsample(residual)

        out = out + residual
        out = self.ReLU(out)
        return out


class Conv1X(nn.Module):
    def __init__(self, use_checkpoint):
        super(Conv1X, self).__init__()
        self.use_checkpoint = use_checkpoint
        self.convBlock = nn.Sequential(  # 1*120*120 -> 64*116*116 -> 64*112*112 -> 64*56*565
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(5, 5), stride=(1, 1), padding=0, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(5, 5), stride=(1, 1), padding=0, bias=False),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        # print("Conv1X : ")
        # print("conv1 : ", self.convBlock[0].state_dict())
        # print("conv2 : ", self.convBlock[3].state_dict())
        if self.use_checkpoint:
            x = checkpoint_sequential(self.convBlock, 2, x)
            # print("Conv1X DONE")
            return x
        else:
            x = self.convBlock(x)
            return x


class Conv2X(nn.Module):
    def __init__(self, use_checkpoint):
        super(Conv2X, self).__init__()
        self.use_checkpoint = use_checkpoint
        self.convBlock = nn.Sequential(
            BottleneckBlock(in_channels=64, hidden_channels=64, out_channels=256),
            BottleneckBlock(in_channels=256, hidden_channels=64, out_channels=256),
            # BottleneckBlock(in_channels=256, hidden_channels=64, out_channels=256)
        )

    def forward(self, x):
        if self.use_checkpoint:
            x = checkpoint_sequential(self.convBlock, 2, x)
            return x
        else:
            x = self.convBlock(x)
            return x


class Conv3X(nn.Module):
    def __init__(self, use_checkpoint):
        super(Conv3X, self).__init__()
        self.use_checkpoint = use_checkpoint
        self.convBlock = nn.Sequential(
            BottleneckBlock(in_channels=256, hidden_channels=128, out_channels=512, stride=2),
            BottleneckBlock(in_channels=512, hidden_channels=128, out_channels=512),
            # BottleneckBlock(in_channels=512, hidden_channels=128, out_channels=512),
            # BottleneckBlock(in_channels=512, hidden_channels=128, out_channels=512)
        )

    def forward(self, x):
        if self.use_checkpoint:
            x = checkpoint_sequential(self.convBlock, 2, x)
            return x
        else:
            x = self.convBlock(x)
            return x


class Conv4X(nn.Module):
    def __init__(self, use_checkpoint):
        super(Conv4X, self).__init__()
        self.use_checkpoint = use_checkpoint
        self.convBlock = nn.Sequential(
            BottleneckBlock(in_channels=512, hidden_channels=256, out_channels=1024, stride=2),
            BottleneckBlock(in_channels=1024, hidden_channels=256, out_channels=1024),
            # BottleneckBlock(in_channels=1024, hidden_channels=256, out_channels=1024),
            # BottleneckBlock(in_channels=1024, hidden_channels=256, out_channels=1024),
            # BottleneckBlock(in_channels=1024, hidden_channels=256, out_channels=1024),
            # BottleneckBlock(in_channels=1024, hidden_channels=256, out_channels=1024)
        )

    def forward(self, x):
        if self.use_checkpoint:
            x = checkpoint_sequential(self.convBlock, 2, x)
            return x
        else:
            x = self.convBlock(x)
            return x


class Conv5X(nn.Module):
    def __init__(self, use_checkpoint):
        super(Conv5X, self).__init__()
        self.use_checkpoint = use_checkpoint
        self.convBlock = nn.Sequential(
            BottleneckBlock(in_channels=1024, hidden_channels=512, out_channels=2048, stride=2),
            BottleneckBlock(in_channels=2048, hidden_channels=512, out_channels=2048),
            # BottleneckBlock(in_channels=2048, hidden_channels=512, out_channels=2048)
        )

    def forward(self, x):
        if self.use_checkpoint:
            x = checkpoint_sequential(self.convBlock, 2, x)
            return x
        else:
            x = self.convBlock(x)
            return x


# ToDo(Alex Han) 添加自动网络生成模块，根据传入参数自动生成四个动态网络模块(conv2_x, conv3_x, conv4_x, conv5_x)
class ResNet(nn.Module):
    def __init__(self, layer_nums: list, use_checkpoint=False):
        # layer_nums = [conv2_x_layer_nums, conv3_x_layer_nums, conv4_x_layer_nums, conv5_x_layer_nums]
        # In ResNet50, layer_nums = [3, 4, 6, 3]
        # layer_nums will be used in future dynamic network generation features
        super(ResNet, self).__init__()
        self.use_checkpoint = use_checkpoint
        self.layer_nums = layer_nums
        self.conv1_x = Conv1X(use_checkpoint=use_checkpoint)
        self.conv2_x = Conv2X(use_checkpoint=use_checkpoint)
        self.conv3_x = Conv3X(use_checkpoint=use_checkpoint)
        self.conv4_x = Conv4X(use_checkpoint=use_checkpoint)
        self.conv5_x = Conv5X(use_checkpoint=use_checkpoint)
        self.avgPool = nn.AvgPool2d(kernel_size=7, stride=1)

    def forward(self, x):
        # x : torch.Size([360, 1, 120, 120])
        # print("ResNet : ")
        x = self.conv1_x(x)
        # print("conv1_x : ", x[0])
        x = self.conv2_x(x)
        # print("conv1_x : ", x[0])
        x = self.conv3_x(x)
        # print("conv1_x : ", x[0])
        x = self.conv4_x(x)
        # print("conv1_x : ", x[0])
        x = self.conv5_x(x)
        # print("conv1_x : ", x[0])
        x = self.avgPool(x)
        # print("avgPool : ", x[0])
        # print("ResNet DONE")
        # x : torch.Size([360, 2048, 1, 1])
        return x


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

        # print("train_module : ")

        # x = x.half()  # Convert input from float32 to float16.
        # print("x : ", x[0])
        # res_output.shape : torch.Size([360, 2048, 1, 1])
        res_output = self.ResNet(x)
        # res_output = torch.rand(360, 2048, 1, 1)
        # res_output.requires_grad = True
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
        # res_output = torch.rand(360, 2048, 1, 1)
        # res_output.requires_grad = False
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
                                 dataLoader: DataLoader, init_weights: bool,
                                 learning_rate: float,
                                 num_epoch: int,
                                 alpha: float,
                                 gamma: float,
                                 device: torch.device):
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

    loss_file = open("../StockPriceForecast/loss.csv", "a")
    loss_file.write("loss\r\n")

    model.apply(xavier_init_weights)
    model.train()
    torch.cuda.empty_cache()
    model.to(device)
    dataLoader.to_device(device)

    attention_weight_seq = []

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # loss_func = dilate_loss  # dilate_loss 计算为负值
    loss_func = nn.MSELoss()
    Loss = []
    # Loss_shape = []
    # Loss_temporal = []
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
            loss = loss_func(model_output, target_data)

            loss.backward()
            optimizer.step()

            Loss.append(loss.sum().cpu().detach())
            # Loss_shape.append(loss_shape.sum().cpu())
            # Loss_temporal.append(loss_temporal.sum().cpu())
            # loss_file.write(str(loss.cpu().detach().numpy()[0]))

            print("loss : ", loss.cpu())
            # print("loss_shape : ", loss_shape.cpu())
            # print("loss_temporal : ", loss_temporal.cpu())
            #
            # del loss, loss_shape, loss_temporal
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
