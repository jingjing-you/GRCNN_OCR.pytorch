#coding:utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F


class BidirectionalLSTM(nn.Module):

    def __init__(self, nIn, nHidden, nOut):
        super(BidirectionalLSTM, self).__init__()

        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)
        self.embedding = nn.Linear(nHidden * 2, nOut)

    def forward(self, input):
        recurrent, _ = self.rnn(input)
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)

        output = self.embedding(t_rec)  # [T * b, nOut]
        output = output.view(T, b, -1)

        return output

class GRCL(nn.Module):
    def __init__(self, in_channels, out_channels, n_iter = 3, kernel_size=3, padding=(1, 1), stride=(1, 1)):
        super(GRCL, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_iter = n_iter

        self.conv_r = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride)
        self.conv_g_r = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1)

        self.conv_f = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride),
                                    nn.BatchNorm2d(out_channels))

        self.conv_g_f = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride),
                                         nn.BatchNorm2d(out_channels))

        self.bn_rec = nn.ModuleList()
        self.bn_gate_rec = nn.ModuleList()
        self.bn_gate_mul = nn.ModuleList()
        for ii in range(n_iter):
            self.bn_rec.append(nn.BatchNorm2d(out_channels))
            self.bn_gate_rec.append(nn.BatchNorm2d(out_channels))
            self.bn_gate_mul.append(nn.BatchNorm2d(out_channels))



    def forward(self, x):
        conv_gate_f = self.conv_g_f(x)
        bn_f = self.conv_f(x)
        x = F.relu(bn_f)

        for ii in range(self.n_iter):
            c_gate_rec = self.bn_gate_rec[ii](self.conv_g_r(x))
            gate = F.sigmoid(conv_gate_f + c_gate_rec)

            c_rec = self.bn_rec[ii](self.conv_r(x))
            x = F.relu(bn_f + self.bn_gate_mul[ii](c_rec*gate))

        return x

class GRCNN(nn.Module):
    def __init__(self, n_class=37):
        super(GRCNN, self).__init__()
        self.n_class = n_class
        self.conv_layer_1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
                                          nn.BatchNorm2d(64), nn.ReLU())
        self.GRCL_layer_1 = GRCL(64, 64, kernel_size=3, stride=(1, 1), padding=(1, 1))
        self.GRCL_layer_2 = GRCL(64, 128, kernel_size=3, stride=(1, 1), padding=(1, 1))
        self.GRCL_layer_3 = GRCL(128, 256, kernel_size=3, stride=(1, 1), padding=(1, 1))

        self.conv_layer_2 = nn.Sequential(nn.Conv2d(256, 512, kernel_size=2, stride=1, padding=0),
                                          nn.BatchNorm2d(512), nn.ReLU())
        self.rnn = nn.Sequential(
            BidirectionalLSTM(512, 256, 256),
            BidirectionalLSTM(256, 256, self.n_class))

    def forward(self, x):
        x = self.conv_layer_1(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = self.GRCL_layer_1(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = self.GRCL_layer_2(x)
        x = F.max_pool2d(x, kernel_size=2, stride=(2, 1), padding=(0, 1))
        x = self.GRCL_layer_3(x)
        x = F.max_pool2d(x, kernel_size=2, stride=(2, 1), padding=(0, 1))
        conv = self.conv_layer_2(x)

        b, c, h, w = conv.size()
        assert h == 1, "the height of conv must be 1"
        conv = conv.squeeze(2)
        conv = conv.permute(2, 0, 1)  # [w, b, c]
        conv = self.rnn(conv)
        return conv




if __name__ == '__main__':
    model = GRCNN(37)
    x = torch.rand(1, 1, 32, 100)
    y = model(x)
    # model = GRCL(32, 64, n_iter=3, kernel_size=3, stride=1, padding=1)
    # x = torch.rand(1, 32, 32, 200)
    # y = model(x)
    #print(model)



