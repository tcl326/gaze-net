import os
import torch
import torch.utils.model_zoo as model_zoo
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import torchvision.models as models
import torch.nn as nn
import torch.nn.parallel
import numpy as np
import pickle as pkl


class FeatureExtractor(nn.Module):
    '''
    call the CNN model as a feature extractor
    '''
    def __init__(self, arch):
        super(FeatureExtractor, self).__init__()
        self.arch = arch   # 'alexnet' or 'vgg16'
        self.features = self.load_pretrained_model()

    def forward(self, img):
        feat = self.features(img)
        return feat

    def load_pretrained_model(self):
        pretrained_model_name = 'pretrained_' + self.arch + '.pkl'
        print("Load model from torchvision.models")
        pretrained_model = models.__dict__[self.arch](pretrained=True)
        pretrained_model = pretrained_model.features    # only keep the conv layers
        return pretrained_model

class SpatialAttentionLayer(nn.Module):
    '''
    compute the spatial attention for each frame
    '''
    def __init__(self, lstm_hidden_size, cnn_feat_size, projected_size):
        '''
        lstm_hidden_size: number of LSTM hidden nodes(LSTM for gaze sequence)
        cnn_feat_size: number of CNN features for each region
        projected_size: output feature size
        '''
        super(SpatialAttentionLayer, self).__init__()
        self.lstm_hidden_size = lstm_hidden_size
        self.cnn_feat_size = cnn_feat_size
        self.projected_size = projected_size
        self.linear_lstm = nn.Linear(lstm_hidden_size, projected_size)
        self.linear_cnn = nn.Linear(cnn_feat_size, projected_size)
        self.linear_weight = nn.Linear(projected_size, 1, bias=False)

    def forward(self, lstm_hidden, cnn_feat):
        '''
        lstm_hidden: (bs, lstm_hidden_size)
        cnn_feat: (bs, cnn_feat_size, sqrt(grid_num), sqrt(grid_num))
        zi = wh * tanh(Wv * V + Wg * H)
        weight : ai = sigmoid(zi)
        '''
        bs = cnn_feat.size()[0]         #normally should be 1
        # 7*7=49, corresponding to the size of CNN output feature size
        cnn_feat = cnn_feat.view(bs, self.cnn_feat_size, -1) # (bs, cnn_feat_size, grid_num)
        grid_num = cnn_feat.size()[2]
        print("grid num")
        print(grid_num)
        weight = []
        for i in range(grid_num):
            H = self.linear_lstm(lstm_hidden)
            V = self.linear_cnn(cnn_feat[:,:,i])
            feat_sum = H + V
            feat_sum = F.tanh(feat_sum)                 # (bs, projected_size)
            feat_sum = self.linear_weight(feat_sum)     # (bs, 1)
            weight.append(feat_sum)
        weight = torch.cat(weight, dim=0)               # (bs * grid_num)
        spatial_weight = F.softmax(weight.view(bs, grid_num))
        return spatial_weight

class TemporalAttentionLayer(nn.Module):
    '''
    compute the temporal attention for each sequence
    '''
    def __init__(self, lstm_hidden_size, spatial_feat_size, projected_size):
        '''
        lstm_hidden_size: number of LSTM hidden nodes(LSTM for image feature sequence)
        spatial_feat_size: number of features from spatial attention layer
        projected_size: output feature size
        '''
        super(SpatialAttentionLayer, self).__init__()
        self.lstm_hidden_size = lstm_hidden_size
        self.spatial_feat_size = spatial_feat_size
        self.projected_size = projected_size
        self.linear_lstm = nn.Linear(lstm_hidden_size, projected_size)
        self.linear_spatial_feat = nn.Linear(spatial_feat_size, projected_size)
        self.linear_weight = nn.Linear(projected_size, 1, bias=False)

    def forward(self, lstm_hidden, spatial_feat):
        '''
        lstm_hidden: (bs, lstm_hidden_size)
        spatial_feat: (bs, time_step, spatial_feat_size)
        zi = wh * tanh(Wv * V + Wg * H)
        weight : ai = sigmoid(zi)
        '''
        bs = spatial_feat.size()[0]         #normally should be 1
        time_step = spatial_feat.size()[1]
        weight = []
        for i in range(time_step):
            H = self.linear_lstm(lstm_hidden)
            V = self.linear_spatial_feat(spatial_feat[:,i,:])
            feat_sum = H + V
            feat_sum = F.tanh(feat_sum)                 # (bs, projected_size)
            feat_sum = self.linear_weight(feat_sum)     # (bs, 1)
            weight.append(feat_sum)
        weight = torch.cat(weight, dim=0)               # (bs * grid_num)
        temporal_weight = F.softmax(weight.view(bs, time_step))
        return temporal_weight
