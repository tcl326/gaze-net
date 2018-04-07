import os
import torch
import torch.utils.model_zoo as model_zoo
from torch.nn.parameter import Parameter
import torchvision.models as models
import torch.nn as nn
import torch.nn.parallel
import numpy as np

from model import *
from util import *


def main():
    bs = 2
    ts = 32
    # set up the feature extractor
    arch = 'alexnet'
    # arch = 'vgg16'
    extractor_model = FeatureExtractor(arch=arch)
    extractor_model.features = torch.nn.DataParallel(extractor_model.features)
    # extractor_model.cuda()      # uncomment this line if using cpu
    extractor_model.eval()

    # pre-processing for the image, should add data generator loader here
    imgs = 255 * np.random.rand(bs, ts, 224, 224, 3)
    imgs = normalize(imgs)

    # compute the features for a batch of images, [bs, 256, 6, 6]
    for i in range(bs):
        img = imgs[i]   #(ts, 3, 224, 224)
        img_var = torch.autograd.Variable(torch.Tensor(img))
        cnn_feat_var = extractor_model(img_var)
        print("cnn fetaure extractor")
        print(cnn_feat_var)

    # test spatial attention layer
    spatial_attention_layer = SpatialAttentionLayer(lstm_hidden_size = 64,
                                cnn_feat_size = 256, projected_size = 64)
    lstm_hidden = np.zeros((bs, 64))
    lstm_hidden_var = torch.autograd.Variable(torch.Tensor(lstm_hidden))
    spatial_weight = spatial_attention_layer(lstm_hidden_var, cnn_feat_var)
    print("spatial weight")
    print(spatial_weight)



if __name__ == '__main__':
    main()
