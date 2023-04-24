import torch
import torch.nn as nn
from torch.nn import functional as F


class PredictionNetwork(nn.Module):
    '''
        Predict class scores and bounding boxes (predict offsets)
    '''

    def __init__(self, num_classes):
        super(PredictionNetwork, self).__init__()

        self.num_classes = num_classes

        # num of prior-boxes in location in feature map
        num_boxes = {"conv4_3": 4, "conv7": 6, "conv8_2": 6,
                     "conv9_2": 6, "conv10_2": 4, "conv11_2": 4}

        # Localization predict offset layer
        self.conv_4_3_loc = nn.Conv2d(512, num_boxes["conv4_3"] * 4,
                                      kernel_size=(3, 3), padding=1)
        self.conv7_loc = nn.Conv2d(1024, num_boxes["conv7"] * 4,
                                   kernel_size=(3, 3), padding=1)
        self.conv8_2_loc = nn.Conv2d(512, num_boxes["conv8_2"]*4,
                                     kernel_size=(3, 3), padding=1)
        self.conv9_2_loc = nn.Conv2d(256, num_boxes["conv9_2"]*4,
                                     kernel_size=(3, 3), padding=1)
        self.conv10_2_loc = nn.Conv2d(256, num_boxes["conv10_2"]*4,
                                      kernel_size=(3, 3), padding=1)
        self.conv11_2_loc = nn.Conv2d(256, num_boxes["conv11_2"]*4,
                                      kernel_size=(3, 3), padding=1)

        # Class predict layer in localization box
        self.conv4_3_cls = nn.Conv2d(512, num_boxes["conv4_3"] * num_classes,
                                     kernel_size=(3, 3), padding=1)
        self.conv7_cls = nn.Conv2d(1024, num_boxes["conv7"] * num_classes,
                                   kernel_size=(3, 3), padding=1)
        self.conv8_2_cls = nn.Conv2d(512, num_boxes["conv8_2"] * num_classes,
                                     kernel_size=(3, 3), padding=1)
        self.conv9_2_cls = nn.Conv2d(256, num_boxes["conv9_2"] * num_classes,
                                     kernel_size=(3, 3), padding=1)
        self.conv10_2_cls = nn.Conv2d(256, num_boxes["conv10_2"] * num_classes,
                                      kernel_size=(3, 3), padding=1)
        self.conv11_2_cls = nn.Conv2d(256, num_boxes["conv11_2"] * num_classes,
                                      kernel_size=(3, 3), padding=1)

        self.weights_init()

    def forward(self, conv4_3_out, conv7_out, conv8_2_out, conv9_2_out,
                conv10_2_out, conv11_2_out):
        '''
            Forward propagation
            Input: feature map in conv layer
        '''
        batch_size = conv4_3_out.size(0)

        l_conv4_3 = self.conv_4_3_loc(conv4_3_out)  # (N, 16, 32, 32)
        l_conv4_3 = l_conv4_3.permute(
            0, 2, 3, 1).contiguous()  # (N, 32, 32, 16)
        l_conv4_3 = l_conv4_3.view(batch_size, -1, 4)  # (N, 4096, 4)

        assert l_conv4_3.size(1) == 4096

        l_conv7 = self.conv7_loc(conv7_out)  # (N, 24, 16, 16)
        l_conv7 = l_conv7.permute(0, 2, 3, 1).contiguous()  # (N, 16, 16, 24)
        l_conv7 = l_conv7.view(batch_size, -1, 4)  # (N, 1536, 4)
        assert l_conv7.size(1) == 1536

        l_conv8_2 = self.conv8_2_loc(conv8_2_out)  # (N, 24, 8, 8)
        l_conv8_2 = l_conv8_2.permute(
            0, 2, 3, 1).contiguous()  # (N, 8, 8, 24)
        l_conv8_2 = l_conv8_2.view(batch_size, -1, 4)  # (N, 384, 4)
        
        assert l_conv8_2.size(1) == 384

        l_conv9_2 = self.conv9_2_loc(conv9_2_out)  # (N, 24, 4, 4)
        l_conv9_2 = l_conv9_2.permute(0, 2, 3, 1).contiguous()  # (N, 4, 4, 24)
        l_conv9_2 = l_conv9_2.view(batch_size, -1, 4)  # (N, 96, 4)
        assert l_conv9_2.size(1) == 96

        l_conv10_2 = self.conv10_2_loc(conv10_2_out)  # (N, 16, 2, 2)
        l_conv10_2 = l_conv10_2.permute(
            0, 2, 3, 1).contiguous()  # (N, 2, 2, 16)
        l_conv10_2 = l_conv10_2.view(batch_size, -1, 4)  # (N, 16, 4)
        assert l_conv10_2.size(1) == 16

        l_conv11_2 = self.conv11_2_loc(conv11_2_out)  # (N, 16, 1, 1)
        l_conv11_2 = l_conv11_2.permute(
            0, 2, 3, 1).contiguous()  # (N, 1, 1, 16)
        l_conv11_2 = l_conv11_2.view(batch_size, -1, 4)  # (N, 4, 4)
        assert l_conv11_2.size(1) == 4

        # Predict class in loc boxes
        c_conv4_3 = self.conv4_3_cls(conv4_3_out)  # (N, 4*classes, 32, 32)
        c_conv4_3 = c_conv4_3.permute(
            0, 2, 3, 1).contiguous()  # (N, 32, 32, 4*classes)
        c_conv4_3 = c_conv4_3.view(
            batch_size, -1, self.num_classes)  # (N, 4096, classes )
        assert c_conv4_3.size(1) == 4096

        c_conv7 = self.conv7_cls(conv7_out)  # (N, 6*classes, 16, 16)

        # (N, 16, 16, 6*classes)
        c_conv7 = c_conv7.permute(0, 2, 3, 1).contiguous()

        # (N, 1536, classes)
        c_conv7 = c_conv7.view(batch_size, -1, self.num_classes)
        assert c_conv7.size(1) == 1536

        c_conv8_2 = self.conv8_2_cls(conv8_2_out)  # (N, 6*clases, 8, 8)
        c_conv8_2 = c_conv8_2.permute(
            0, 2, 3, 1).contiguous()  # (N, 8, 8, 6*classes)
        c_conv8_2 = c_conv8_2.view(
            batch_size, -1, self.num_classes)  # (N, 384, classes)
        assert c_conv8_2.size(1) == 384

        c_conv9_2 = self.conv9_2_cls(conv9_2_out)  # (N, 6*classes, 4, 4)
        c_conv9_2 = c_conv9_2.permute(
            0, 2, 3, 1).contiguous()  # (N, 4, 4, 6*classes)
        c_conv9_2 = c_conv9_2.view(
            batch_size, -1, self.num_classes)  # (N, 96, classes)
        assert c_conv9_2.size(1) == 96

        c_conv10_2 = self.conv10_2_cls(conv10_2_out)  # (N, 4*classes, 2, 2)
        c_conv10_2 = c_conv10_2.permute(
            0, 2, 3, 1).contiguous()  # (N, 2, 2, 4*classes)
        c_conv10_2 = c_conv10_2.view(
            batch_size, -1, self.num_classes)  # (N, 16, classes)
        assert c_conv10_2.size(1) == 16

        c_conv11_2 = self.conv11_2_cls(conv11_2_out)  # (N, 4*classes, 1, 1)
        c_conv11_2 = c_conv11_2.permute(
            0, 2, 3, 1).contiguous()  # (N, 1, 1, 4*classes)
        c_conv11_2 = c_conv11_2.view(
            batch_size, -1, self.num_classes)  # (N, 4, classes)
        assert c_conv11_2.size(1) == 4

        locs_pred = torch.cat([l_conv4_3, l_conv7, l_conv8_2, l_conv9_2,
                               l_conv10_2, l_conv11_2], dim=1)  # (N, 6132, 4)
        assert locs_pred.size(0) == batch_size
        assert locs_pred.size(1) == 6132
        assert locs_pred.size(2) == 4

        cls_pred = torch.cat([c_conv4_3, c_conv7, c_conv8_2, c_conv9_2,
                              c_conv10_2, c_conv11_2], dim=1)  # (N, 6132, classes)
        assert cls_pred.size(0) == batch_size
        assert cls_pred.size(1) == 6132
        assert cls_pred.size(2) == self.num_classes

        return locs_pred, cls_pred

    def weights_init(self):
        '''
            Init conv parameters by xavier
        '''
        for c in self.children():
            if isinstance(c, nn.Conv2d):
                nn.init.xavier_uniform_(c.weight)
                nn.init.zeros_(c.bias)
