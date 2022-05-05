import winnt
from libs.modules.includes import *
from libs.modules.thirdPart import *

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
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1,
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
            nn.Conv2d(in_channels=in_channels, out_channels=hidden_channels, kernel_size=1, stride=1,
                      padding=0, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=3, stride=stride,
                      padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU()
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_channels, out_channels=out_channels, kernel_size=1, stride=1,
                      padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

        # ToDo(Alex Han) 在自动网络生成功能完成时，需要将这里的 downsample 模块放到自动网络生成模块中，并从 downSample 传入
        self.downsample = downsample
        if in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride,
                          # downSample
                          padding=0, bias=False),
                nn.BatchNorm2d(out_channels)
            )

        self.ReLU = nn.ReLU()

    def forward(self, x: torch.Tensor):
        residual = x
        out = self.conv3(self.conv2(self.conv1(x)))

        print("out : ", out.shape)
        if self.downsample is not None:
            print("Do downsample!")
            residual = self.downsample(residual)
            print("downsample : ", residual.shape)

        out = out + residual
        out = self.ReLU(out)
        return out


class Conv2X(nn.Module):
    def __init__(self):
        super(Conv2X, self).__init__()
        self.convBlock0 = BottleneckBlock(in_channels=64, hidden_channels=64, out_channels=256)
        self.convBlock1 = BottleneckBlock(in_channels=256, hidden_channels=64, out_channels=256)
        self.convBlock2 = BottleneckBlock(in_channels=256, hidden_channels=64, out_channels=256)

    def forward(self, x):
        x = self.convBlock0(x)
        x = self.convBlock1(x)
        x = self.convBlock2(x)
        return x


class Conv3X(nn.Module):
    def __init__(self):
        super(Conv3X, self).__init__()
        self.convBlock0 = BottleneckBlock(in_channels=256, hidden_channels=128, out_channels=512,
                                          stride=2)  # maxPooling or AvgPooling
        self.convBlock1 = BottleneckBlock(in_channels=512, hidden_channels=128, out_channels=512)
        self.convBlock2 = BottleneckBlock(in_channels=512, hidden_channels=128, out_channels=512)
        self.convBlock3 = BottleneckBlock(in_channels=512, hidden_channels=128, out_channels=512)

    def forward(self, x):
        x = self.convBlock0(x)
        x = self.convBlock1(x)
        x = self.convBlock2(x)
        x = self.convBlock3(x)
        return x


class Conv4X(nn.Module):
    def __init__(self):
        super(Conv4X, self).__init__()
        self.convBlock0 = BottleneckBlock(in_channels=512, hidden_channels=256, out_channels=1024, stride=2)
        self.convBlock1 = BottleneckBlock(in_channels=1024, hidden_channels=256, out_channels=1024)
        self.convBlock2 = BottleneckBlock(in_channels=1024, hidden_channels=256, out_channels=1024)
        self.convBlock3 = BottleneckBlock(in_channels=1024, hidden_channels=256, out_channels=1024)
        self.convBlock4 = BottleneckBlock(in_channels=1024, hidden_channels=256, out_channels=1024)
        self.convBlock5 = BottleneckBlock(in_channels=1024, hidden_channels=256, out_channels=1024)

    def forward(self, x):
        x = self.convBlock0(x)
        x = self.convBlock1(x)
        x = self.convBlock2(x)
        x = self.convBlock3(x)
        x = self.convBlock4(x)
        x = self.convBlock5(x)
        return x


class Conv5X(nn.Module):
    def __init__(self):
        super(Conv5X, self).__init__()
        self.convBlock0 = BottleneckBlock(in_channels=1024, hidden_channels=512, out_channels=2048, stride=2)
        self.convBlock1 = BottleneckBlock(in_channels=2048, hidden_channels=512, out_channels=2048)
        self.convBlock2 = BottleneckBlock(in_channels=2048, hidden_channels=512, out_channels=2048)

    def forward(self, x):
        x = self.convBlock0(x)
        x = self.convBlock1(x)
        x = self.convBlock2(x)
        return x


# ToDo(Alex Han) 添加自动网络生成模块，根据传入参数自动生成四个动态网络模块(conv2_x, conv3_x, conv4_x, conv5_x)
class ResNet(nn.Module):
    def __init__(self, use_checkpoint=False):
        super(ResNet, self).__init__()
        self.use_checkpoint = use_checkpoint
        self.conv1_x = nn.Sequential(  # 1*120*120 -> 64*116*116 -> 64*112*112 -> 64*56*56
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=5, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv2_x = Conv2X()
        self.conv3_x = Conv3X()
        self.conv4_x = Conv4X()
        self.conv5_x = Conv5X()
        self.avgPool = nn.AvgPool2d(kernel_size=7, stride=1)

    def forward(self, x):
        if self.use_checkpoint:
            # 使用checkpoint和checkpoint_sequential降低显存占用
            x = checkpoint_sequential(self.conv1_x, 2, x)
            x = checkpoint(self.conv2_x, x)
            x = checkpoint(self.conv3_x, x)
            x = checkpoint(self.conv4_x, x)
            x = checkpoint(self.conv5_x, x)  # [360, 2048, 7, 7]
            x = checkpoint(self.avgPool, x)  # [360, 2048, 1, 1]
            return x
        else:
            x = self.conv1_x(x)
            x = self.conv2_x(x)
            x = self.conv3_x(x)
            x = self.conv4_x(x)
            x = self.conv5_x(x)
            x = self.avgPool(x)
            return x


class Seq2SeqEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.1):
        super(Seq2SeqEncoder, self).__init__()
        self.embedding = nn.Linear(input_size, hidden_size)
        self.gru = nn.GRU(input_size=hidden_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout)

    def forward(self, x):
        x = self.embedding(x)
        out, hidden_state = self.gru(x)
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
    def __init__(self, hidden_size, output_size, num_layers, dropout=0.1):
        super(Seq2SeqAttentionDecoder, self).__init__()
        self.embedding = nn.Linear(hidden_size, hidden_size)
        self.attention = AdditiveAttention(key_size=hidden_size, query_size=hidden_size,
                                           num_hiddens=hidden_size, dropout=dropout)
        self.gru = nn.GRU(input_size=hidden_size * 2, hidden_size=hidden_size,
                          num_layers=num_layers, dropout=dropout)
        self.dense = nn.Linear(hidden_size, output_size)

        self._attention_weights = []

    def init_state(self, enc_outputs, enc_valid_lens=None, *args):
        # outputs的形状为(batch_size，num_steps，num_hiddens).
        # hidden_state的形状为(num_layers，batch_size，num_hiddens)
        outputs, hidden_state = enc_outputs
        return outputs.permute(1, 0, 2), hidden_state, enc_valid_lens

    def forward(self, _X, state):
        enc_outputs, hidden_state, enc_valid_lens = state
        _X = self.embedding(_X)
        outputs, self._attention_weights = [], []
        for x in _X:
            query = torch.unsqueeze(hidden_state[-1], dim=1)
            context = self.attention(query, enc_outputs, enc_outputs, enc_valid_lens)  #
            # print("context : ", context.shape)
            # print("torch.unsqueeze(x, dim=1) : ", torch.unsqueeze(x, dim=1).shape)
            x = torch.cat([context, torch.unsqueeze(x, dim=1)], dim=-1)
            # print("x : ", x.shape)
            # print("x.permute(1, 0, 2) : ", x.permute(1, 0, 2).shape)
            out, hidden_state = self.gru(x.permute(1, 0, 2), hidden_state)
            outputs.append(out)
            self._attention_weights.append(self.attention.attention_weights)
        outputs = self.dense(torch.cat(outputs, dim=0))
        return outputs.permute(1, 0, 2), [enc_outputs, hidden_state, enc_valid_lens]

    @property
    def attention_weights(self):
        return self._attention_weights


class EncoderDecoder(nn.Module):
    def __init__(self, Encoder, Decoder):
        super(EncoderDecoder, self).__init__()
        self.encoder = Encoder
        self.decoder = Decoder

    def forward(self, enc_X, dec_X, *args):
        enc_outputs = self.encoder(enc_X, *args)
        dec_state = self.decoder.init_state(enc_outputs, *args)
        return self.decoder(dec_X, dec_state)


class Seq2Seq(nn.Module):
    def __init__(self):
        super(Seq2Seq, self).__init__()
        self.encoder = Seq2SeqEncoder()
        self.decoder = Seq2SeqDecoder()

    def forward(self, x):
        pass


class Seq2SeqAttention(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, dropout=0.1):
        super(Seq2SeqAttention, self).__init__()
        self.encoder = Seq2SeqEncoder(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout)
        self.decoder = Seq2SeqAttentionDecoder(hidden_size=hidden_size, output_size=output_size, num_layers=num_layers, dropout=dropout)

    def forward(self, enc_X, dec_X, *args):
        enc_outputs = self.encoder(enc_X, *args)
        dec_state = self.decoder.init_state(enc_outputs, *args)
        return self.decoder(dec_X, dec_state)


# ToDo(Alex Han) 需要传入参数
class ResNetSeq2Seq(nn.Module):
    def __init__(self, use_checkpoint=False):
        super(ResNetSeq2Seq, self).__init__()
        self.ResNet = ResNet(use_checkpoint=use_checkpoint)
        self.Seq2Seq = Seq2SeqAttention(input_size=2048, hidden_size=2048, output_size=1440, num_layers=2, dropout=0.1)

    def forward(self, x):
        x = self.ResNet(x)
        x = x.reshape(360, 1, 2048)
        x = self.Seq2Seq(x)
        # 预测函数
        return x


def train_seq2seq(module, dataLoader, learning_rate, num_epoch,  device):

    for epoch in num_epoch:


        pass

    return


def predict_seq2seq():

    return


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

    encoder = Seq2SeqEncoder(input_size=2048, hidden_size=2048, num_layers=2, dropout=0.1)
    decoder = Seq2SeqAttentionDecoder(hidden_size=2048, output_size=1440, num_layers=2, dropout=0.1)
    encoder.eval()
    decoder.eval()

    X = torch.rand(360, 1, 2048)
    print("X : ", X.shape)
