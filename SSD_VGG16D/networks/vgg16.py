import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from ..utils import colorstr, decimate


class VGG16DBaseNetwork(nn.Module):
    def __init__(self, pretrained=False):
        super(VGG16DBaseNetwork, self).__init__()

        """
        The Conv2d layer is used here, so each image is treated independently, which is different from the original paper.
        The original paper uses the Conv3d layer, which is used to process the video data. 
        The Conv3d layer gives a temporal context to the data, and the said frames of the video are processed together.
        """

        """
        kernel_size: The size of the convolution kernel is 3x3, which is the same as the original paper.
        https://stats.stackexchange.com/questions/296679/what-does-kernel-size-mean
        
        calculate the output size of the convolution layer: 
        https://dingyan89.medium.com/calculating-parameters-of-convolutional-and-fully-connected-layers-with-keras-186590df36c6#:~:text=The%20kernel%20size%20of%20max,5%2C5%2C16).
        """

        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=(3, 3), padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=(3, 3), padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=(3, 3), padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=(3, 3), padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=(3, 3), padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=(3, 3), padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=(2, 2), stride=2, ceil_mode=True)

        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=(3, 3), padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=(3, 3), padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=(3, 3), padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)

        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=(3, 3), padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=(3, 3), padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=(3, 3), padding=1)
        self.pool5 = nn.MaxPool2d(kernel_size=(3, 3), stride=1, padding=1)

        self.conv6 = nn.Conv2d(
            512, 1024, kernel_size=(3, 3), padding=6, dilation=6)
        self.conv7 = nn.Conv2d(1024, 1024, kernel_size=(1, 1))

        if pretrained:
            self.load_pretrained()
        else:
            self.weights_init()

    def forward(self, image):
        '''
            Forward propagation
            image: image, a tensor of dimensions (N, 3, 256, 256)

            Out: feature map conv4_3, conv7
        '''
        # Image Dimensions: (256, 256)
        x = image  # (N, 3, 256, 256)
        x = F.relu(self.conv1_1(x))  # (N, 64, 256, 256)
        x = F.relu(self.conv1_2(x))  # (N, 64, 256, 256)
        x = self.pool1(x)  # (N, 64, 128, 128)

        x = F.relu(self.conv2_1(x))  # (N, 128, 128, 128)
        x = F.relu(self.conv2_2(x))  # (N, 128, 128, 128)
        x = self.pool2(x)  # (N, 128, 64, 64)

        x = F.relu(self.conv3_1(x))  # (N, 256, 64, 64)
        x = F.relu(self.conv3_2(x))  # (N, 256, 64, 64)
        x = F.relu(self.conv3_3(x))  # (N, 256, 64, 64)
        x = self.pool3(x)  # (N, 256, 32, 32)

        x = F.relu(self.conv4_1(x))  # (N, 512, 32, 32)
        x = F.relu(self.conv4_2(x))  # (N, 512, 32, 32)
        x = F.relu(self.conv4_3(x))  # (N, 512, 32, 32)
        conv4_3_out = x  # (N, 512, 32, 32)
        x = self.pool4(x)  # (N, 512, 16, 16)

        x = F.relu(self.conv5_1(x))  # (N, 512, 16, 16)
        x = F.relu(self.conv5_2(x))  # (N, 512, 16, 16)
        x = F.relu(self.conv5_3(x))  # (N, 512, 16, 16)
        x = self.pool5(x)  # (N, 512, 16, 16)

        x = F.relu(self.conv6(x))  # (N, 1024, 16, 16)

        conv7_out = F.relu(self.conv7(x))  # (N, 1024, 16, 16)

        return conv4_3_out, conv7_out

    def load_pretrained(self):
        '''
            Use a VGG-16 pretrained on the ImageNet task for conv1-->conv5
            Convert conv6, conv7 to pretrained
        '''
        print("Loading pretrained base model...")
        state_dict = self.state_dict()
        param_names = list(state_dict.keys())

        pretrained_state_dict = torchvision.models.vgg16(
            pretrained=True).state_dict()
        pretrained_param_names = list(pretrained_state_dict.keys())

        for i, parameters in enumerate(param_names[:26]):   
            state_dict[parameters] = pretrained_state_dict[pretrained_param_names[i]]

        # convert fc6, fc7 in pretrained to conv6, conv7 in model
        fc6_weight = pretrained_state_dict['classifier.0.weight'].view(
            4096, 512, 7, 7)
        fc6_bias = pretrained_state_dict['classifier.0.bias']
        state_dict['conv6.weight'] = decimate(
            fc6_weight, m=[4, None, 3, 3])  # (1024, 512, 3, 3)
        state_dict['conv6.bias'] = decimate(fc6_bias, m=[4])  # (1024)

        fc7_weight = pretrained_state_dict['classifier.3.weight'].view(
            4096, 4096, 1, 1)
        fc7_bias = pretrained_state_dict['classifier.3.bias']
        state_dict['conv7.weight'] = decimate(fc7_weight, m=[4, 4, None, None])
        state_dict['conv7.bias'] = decimate(fc7_bias, m=[4])

        self.load_state_dict(state_dict)
        print(colorstr("Loaded base model", 'green'))

    def weights_init(self):
        '''
            Init conv parameters by xavier
        '''
        for c in self.children():
            if isinstance(c, nn.Conv2d):
                nn.init.xavier_uniform_(c.weight)
                nn.init.zeros_(c.bias)
