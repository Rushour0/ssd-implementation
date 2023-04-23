from torch import nn
from torch.nn import functional as F

class AuxiliaryNetwork(nn.Module):
    '''
        Add auxiliary structure to the network to produce 
        detections with the following key features
    '''
    def __init__(self):
        super(AuxiliaryNetwork, self).__init__()
        
        self.conv8_1 = nn.Conv2d(1024, 256, kernel_size=(1, 1), padding= 0)
        self.conv8_2 = nn.Conv2d(256, 512, kernel_size= (3, 3), padding= 1, stride= 2)
        
        self.conv9_1 = nn.Conv2d(512, 128, kernel_size= (1, 1), padding= 0)
        self.conv9_2 = nn.Conv2d(128, 256, kernel_size= (3, 3), padding= 1, stride= 2)
        
        self.conv10_1 = nn.Conv2d(256, 128, kernel_size= (1, 1), padding= 0)
        self.conv10_2 = nn.Conv2d(128, 256, kernel_size= (3, 3), padding= 0)
        
        self.conv11_1 = nn.Conv2d(256, 128, kernel_size= (1, 1), padding= 0)
        self.conv11_2 = nn.Conv2d(128, 256, kernel_size= (3, 3), padding= 0)
        
        self.weights_init()
    
    def forward(self, conv7_out):
        '''
            Forward propagation
            conv7_out: feature map from basemodel VGG-16, tensor of 
            dimensions of (N, 1024, 19, 19)
            
            Out: feature map conv8_2, conv9_2, conv10_2, conv11_2
        '''
        x = conv7_out    #(N, 1024, 19, 19)
        x = F.relu(self.conv8_1(x))    #(N, 256, 19, 19)
        x = F.relu(self.conv8_2(x))    #(N, 512, 10, 10)
        conv8_2_out = x
        
        x = F.relu(self.conv9_1(x))    #(N, 128, 10, 10)
        x = F.relu(self.conv9_2(x))    #(N, 256, 5, 5)
        conv9_2_out = x
        
        x = F.relu(self.conv10_1(x))   #(N, 128, 5, 5)
        x = F.relu(self.conv10_2(x))   #(N, 256, 3, 3)
        conv10_2_out = x
        
        x = F.relu(self.conv11_1(x))   #(N, 128, 3, 3)
        conv11_2_out = F.relu(self.conv11_2(x))   #(N, 256, 1, 1)
        
        return conv8_2_out, conv9_2_out, conv10_2_out, conv11_2_out
    
    def weights_init(self):
        '''
            Init conv parameters by xavier
        '''
        for c in self.children():
            if isinstance(c, nn.Conv2d):
                nn.init.xavier_uniform_(c.weight)
                nn.init.zeros_(c.bias)
