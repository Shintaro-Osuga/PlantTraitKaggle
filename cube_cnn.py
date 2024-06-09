import torch.nn as nn
import torch
import torch.nn.functional as F
from KANlayer import *
from KAN import *
#cube_padding = (33*33) - cube_size*cube_size
#input dims: (6(num_sides), cub_size+cube_padding(6,33*2,33*3), max_move_num(99))
#output dims: (1, num_moves+move_padding)
class cube_CNN_1(nn.Module):
    def __init__(self, num_classes, aux_num_classes, feature_cols):
        super(cube_CNN_1, self).__init__()
        
        self.sqr5_conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3,3), padding=(7,7))
        #batch32
        self.pool3 = nn.AvgPool2d((3,3), stride=3)
        self.sqr5_conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(5,5), padding=(5,5))
        
        self.sqr5_conv3 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(3,3), padding=(6,6))
        #bacth32
        #pool3
        self.sqr5_conv4 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(1,1))
        
        self.sqr5_conv5 = nn.Conv2d(in_channels=16, out_channels=64, kernel_size=(5,5), padding=(8,8))
        #batch64
        #pool3
        self.sqr5_conv6 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(1,1))
        
        self.sqr5_conv7 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(3,3), padding=(8,8))
        #batch16
        self.pool5 = nn.AvgPool2d((5,5), stride=3)
        self.sqr5_conv8 = nn.Conv2d(in_channels=16, out_channels=64, kernel_size=(1,1))
        
        self.sqr5_conv9 = nn.Conv2d(in_channels=64, out_channels=256, kernel_size=(5,5), padding=(10,10))
        #batch256
        #pool5
        self.sqr5_conv10 = nn.Conv2d(in_channels=256, out_channels=2048, kernel_size=(1,1))
        
        self.sqr5_conv11 = nn.Conv2d(in_channels=2048, out_channels=4096, kernel_size=(3,3), padding=(6,6))
        #batch4096
        #pool3
        self.sqr5_conv12 = nn.Conv2d(in_channels=4096, out_channels=2048, kernel_size=(1,1))
        
        self.sqr5_conv13 = nn.Conv2d(in_channels=2048, out_channels=256, kernel_size=(5,5), padding=(10,10))
        #batch256
        #pool5
        self.sqr5_conv14 = nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=(1,1))
        
        self.sqr5_conv15 = nn.Conv2d(in_channels=1024, out_channels=2048, kernel_size=(3,3), padding=(6,6))
        #bacth2048
        #pool3
        self.sqr5_conv16 = nn.Conv2d(in_channels=2048, out_channels=256, kernel_size=(1,1))
        
        self.sqr5_conv17 = nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=(5,5), padding=(8,8))
        #batch1024
        #pool3
        self.sqr5_conv18 = nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=(1,1))
        
        self.sqr5_conv19 = nn.Conv2d(in_channels=256, out_channels=2048, kernel_size=(1,1))
        self.sqr5_conv20 = nn.Conv2d(in_channels=2048, out_channels=256, kernel_size=(1,1), padding=(1,1))
        #batch256
        #pool3
        self.sqr5_conv21 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=(1,1))
        self.sqr5_conv22 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(1,1))
        
        self.batchn16 = nn.BatchNorm2d(16)
        self.batchn32 = nn.BatchNorm2d(32)
        self.batchn64 = nn.BatchNorm2d(64)
        self.batchn128 = nn.BatchNorm2d(128)
        self.batchn256 = nn.BatchNorm2d(256)
        self.batchn1024 = nn.BatchNorm2d(1024)
        self.btachn2048 = nn.BatchNorm2d(2048)
        self.btachn4096 = nn.BatchNorm2d(4096)
        # self.batchn4 = nn.BatchNorm2d(1)
        # self.batchn5 = nn.BatchNorm2d(1)
        in_size = 256 #163
        self.out_conv_nn1 = nn.Linear(in_size, 256)
        self.out_conv_nn2 = nn.Linear(256, 64)
        self.out_conv_nn3 = nn.Linear(64,32)
        self.out_conv_nn4 = nn.Linear(32, 163)
        
        
        self.conv_dropout = nn.Dropout(0.1)
        #input for this layer will be torch.concat(horiz_conv, vert_conv.T)
        # self.inconv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=(4,4))
        
        self.conv_feat1 = nn.Linear(163*2, 256)
        self.batchnn1d1 = nn.BatchNorm1d(256)
        self.conv_feat2 = nn.Linear(256, 32)
        self.batchnn1d2 = nn.BatchNorm1d(32)
        self.conv_feat_out = nn.Linear(32, 6)
        self.conv_feat_out_aux = nn.Linear(32, 6)
        
        self.fc_dropout = nn.Dropout(0.1)
        
        
    def forward(self, img, feature):

        
        sqr5_conv = self.conv_dropout(self.pool3(self.batchn32(self.sqr5_conv1(img))))
        sqr5_conv = self.conv_dropout(self.pool5(self.sqr5_conv2(sqr5_conv)))

        sqr5_conv = self.pool3(self.batchn32(self.sqr5_conv3(sqr5_conv)))
        sqr5_conv = self.conv_dropout(self.sqr5_conv4(sqr5_conv))
        
        sqr5_conv = self.pool3(self.batchn64(self.sqr5_conv5(sqr5_conv)))
        sqr5_conv = self.conv_dropout(self.sqr5_conv6(sqr5_conv))

        sqr5_conv = self.pool5(self.batchn16(self.sqr5_conv7(sqr5_conv)))
        sqr5_conv = self.conv_dropout(self.sqr5_conv8(sqr5_conv))
        
        sqr5_conv = self.pool5(self.batchn256(self.sqr5_conv9(sqr5_conv)))
        sqr5_conv = self.conv_dropout(self.sqr5_conv10(sqr5_conv))
        
        sqr5_conv = self.pool3(self.btachn4096(self.sqr5_conv11(sqr5_conv)))
        sqr5_conv = self.conv_dropout(self.sqr5_conv12(sqr5_conv))
        
        sqr5_conv = self.pool5(self.batchn256(self.sqr5_conv13(sqr5_conv)))
        sqr5_conv = self.conv_dropout(self.sqr5_conv14(sqr5_conv))
        
        sqr5_conv = self.pool3(self.btachn2048(self.sqr5_conv15(sqr5_conv)))
        sqr5_conv = self.conv_dropout(self.sqr5_conv16(sqr5_conv))
        
        sqr5_conv = self.pool3(self.batchn1024(self.sqr5_conv17(sqr5_conv)))
        sqr5_conv = self.conv_dropout(self.sqr5_conv18(sqr5_conv))
        sqr5_conv = self.conv_dropout(self.sqr5_conv19(sqr5_conv))
        
        sqr5_conv = self.pool3(self.batchn256(self.sqr5_conv20(sqr5_conv)))
        sqr5_conv = self.conv_dropout(self.sqr5_conv21(sqr5_conv))
        sqr5_conv = self.conv_dropout(self.sqr5_conv22(sqr5_conv))
        
        # print(sqr5_conv.size())
        b = sqr5_conv.size()[0]
        sqr5_flat = torch.reshape(sqr5_conv, (b,-1))
        lin_conv = self.fc_dropout(self.out_conv_nn1(sqr5_flat))
        lin_conv = self.out_conv_nn2(lin_conv)
        lin_conv = self.fc_dropout(self.out_conv_nn3(lin_conv))
        lin_conv = self.out_conv_nn4(lin_conv)

        conv_feat_x = torch.concat([lin_conv, feature], dim=1)
        
        conv_feat_x = self.fc_dropout(self.batchnn1d1(self.conv_feat1(conv_feat_x)))
        conv_feat_x = self.batchnn1d2(self.conv_feat2(conv_feat_x))
        out = self.conv_feat_out(conv_feat_x)
        aux_out = self.conv_feat_out_aux(conv_feat_x)

        return {'out':out, 'aux_out':aux_out}
    
import torch.nn as nn
import torch
import torch.nn.functional as F
#cube_padding = (33*33) - cube_size*cube_size
#input dims: (6(num_sides), cub_size+cube_padding(6,33*2,33*3), max_move_num(99))
#output dims: (1, num_moves+move_padding)
class cube_CNN_2(nn.Module):
    def __init__(self, num_classes, aux_num_classes, feature_cols):
        super(cube_CNN_2, self).__init__()
        
        self.sqr5_conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3,3), padding=(7,7))
        #batch32
        self.pool3 = nn.AvgPool2d((3,3), stride=3)
        self.sqr5_conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(5,5), padding=(5,5))
        
        self.sqr5_conv3 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(3,3), padding=(6,6))
        #bacth32
        #pool3
        self.sqr5_conv4 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(1,1))
        
        self.sqr5_conv5 = nn.Conv2d(in_channels=16, out_channels=64, kernel_size=(5,5), padding=(8,8))
        #batch64
        #pool3
        self.sqr5_conv6 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(1,1))
        
        self.sqr5_conv7 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(3,3), padding=(8,8))
        #batch16
        self.pool5 = nn.AvgPool2d((5,5), stride=3)
        self.sqr5_conv8 = nn.Conv2d(in_channels=16, out_channels=64, kernel_size=(1,1))
        
        self.sqr5_conv9 = nn.Conv2d(in_channels=64, out_channels=256, kernel_size=(5,5), padding=(10,10))
        #batch256
        #pool5
        
        self.sqr5_conv10 = nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=(1,1))
        self.sqr5_conv11 = nn.Conv2d(in_channels=1024, out_channels=2048, kernel_size=(1,1))
        
        self.sqr5_conv12 = nn.Conv2d(in_channels=2048, out_channels=4096, kernel_size=(1,1))
        self.sqr5_conv13 = nn.Conv2d(in_channels=4096, out_channels=2048, kernel_size=(1,1), padding=(1,1))
        #batch2048
        #pool3
        self.sqr5_conv14 = nn.Conv2d(in_channels=2048, out_channels=128, kernel_size=(1,1))
        self.sqr5_conv15 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(1,1))
        
        self.batchn16 = nn.BatchNorm2d(16)
        self.batchn32 = nn.BatchNorm2d(32)
        self.batchn64 = nn.BatchNorm2d(64)
        self.batchn128 = nn.BatchNorm2d(128)
        self.batchn256 = nn.BatchNorm2d(256)
        self.batchn1024 = nn.BatchNorm2d(1024)
        self.batchn2048 = nn.BatchNorm2d(2048)
        # self.btachn4096 = nn.BatchNorm2d(4096)
        # self.batchn4 = nn.BatchNorm2d(1)
        # self.batchn5 = nn.BatchNorm2d(1)
        in_size = 256 #163
        self.out_conv_nn1 = nn.Linear(in_size, 256)
        self.out_conv_nn2 = nn.Linear(256, 64)
        self.out_conv_nn3 = nn.Linear(64,32)
        self.out_conv_nn4 = nn.Linear(32, 163)
        
        
        self.conv_dropout = nn.Dropout(0.1)
        #input for this layer will be torch.concat(horiz_conv, vert_conv.T)
        # self.inconv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=(4,4))
        
        self.conv_feat1 = nn.Linear(163*2, 256)
        self.batchnn1d1 = nn.BatchNorm1d(256)
        self.conv_feat2 = nn.Linear(256, 32)
        self.batchnn1d2 = nn.BatchNorm1d(32)
        self.conv_feat_out = nn.Linear(32, 6)
        self.conv_feat_out_aux = nn.Linear(32, 6)
        
        self.fc_dropout = nn.Dropout(0.1)
        
        
    def forward(self, img, feature):

        
        sqr5_conv = self.conv_dropout(self.pool3(self.batchn32(self.sqr5_conv1(img))))
        sqr5_conv = self.conv_dropout(self.pool5(self.sqr5_conv2(sqr5_conv)))

        sqr5_conv = self.pool3(self.batchn32(self.sqr5_conv3(sqr5_conv)))
        sqr5_conv = self.conv_dropout(self.sqr5_conv4(sqr5_conv))
        
        sqr5_conv = self.pool3(self.batchn64(self.sqr5_conv5(sqr5_conv)))
        sqr5_conv = self.conv_dropout(self.sqr5_conv6(sqr5_conv))

        sqr5_conv = self.pool5(self.batchn16(self.sqr5_conv7(sqr5_conv)))
        sqr5_conv = self.conv_dropout(self.sqr5_conv8(sqr5_conv))
        
        sqr5_conv = self.pool5(self.batchn256(self.sqr5_conv9(sqr5_conv)))

        sqr5_conv = self.conv_dropout(self.sqr5_conv10(sqr5_conv))
        sqr5_conv = self.conv_dropout(self.sqr5_conv11(sqr5_conv))
        sqr5_conv = self.conv_dropout(self.sqr5_conv12(sqr5_conv))
        
        sqr5_conv = self.pool3(self.batchn2048(self.sqr5_conv13(sqr5_conv)))
        sqr5_conv = self.conv_dropout(self.sqr5_conv14(sqr5_conv))
        sqr5_conv = self.conv_dropout(self.sqr5_conv15(sqr5_conv))
        
        # print(sqr5_conv.size())
        b = sqr5_conv.size()[0]
        sqr5_flat = torch.reshape(sqr5_conv, (b,-1))
        lin_conv = self.fc_dropout(self.out_conv_nn1(sqr5_flat))
        lin_conv = self.out_conv_nn2(lin_conv)
        lin_conv = self.fc_dropout(self.out_conv_nn3(lin_conv))
        lin_conv = self.out_conv_nn4(lin_conv)

        conv_feat_x = torch.concat([lin_conv, feature], dim=1)
        
        conv_feat_x = self.fc_dropout(self.batchnn1d1(self.conv_feat1(conv_feat_x)))
        conv_feat_x = self.batchnn1d2(self.conv_feat2(conv_feat_x))
        out = self.conv_feat_out(conv_feat_x)
        aux_out = self.conv_feat_out_aux(conv_feat_x)

        return {'out':out, 'aux_out':aux_out}
    

class NLWCNN(nn.Module):
    def __init__(self):
        from NonLinWeight import NonLinWeight
        from torch.nn.utils.parametrizations import weight_norm

        super().__init__()
        
        #0.3 model
        # self.model = nn.Sequential(nn.Conv2d(4,16,kernel_size=(3,3), padding='same'),
        #                            nn.ReLU(),
        #                            nn.BatchNorm2d(16),
        #                            nn.Dropout2d(0.1),
        #                            NonLinWeight(function=torch.cos, shape=224, batch_first=True, negative=False),
        #                            nn.AvgPool2d(kernel_size=2),
        #                            nn.Conv2d(16,64,kernel_size=(5,5), padding='same'),
        #                            nn.LeakyReLU(),
        #                            nn.BatchNorm2d(64),
        #                            nn.Dropout2d(0.2),
        #                            NonLinWeight(function=torch.sin, shape=112, batch_first=True, negative=True),
        #                            nn.AvgPool2d(kernel_size=2),
        #                            nn.Conv2d(64,128,kernel_size=(3,3), padding='same'),
        #                            nn.LeakyReLU(),
        #                            nn.BatchNorm2d(128),
        #                            nn.Dropout2d(0.1),
        #                            NonLinWeight(function=torch.sin, shape=56, batch_first=True, negative=False),
        #                            nn.AvgPool2d(kernel_size=2),
        #                            nn.Conv2d(128,32,kernel_size=(9,9), padding='same'),
        #                            nn.LeakyReLU(),
        #                            nn.BatchNorm2d(32),
        #                            nn.Dropout2d(0.2),
        #                            NonLinWeight(function=torch.cos, shape=28, batch_first=True, negative=True),
        #                            nn.AvgPool2d(kernel_size=2),
        #                            nn.Conv2d(32,64,kernel_size=(3,3), padding='same'),
        #                            nn.LeakyReLU(),
        #                            nn.BatchNorm2d(64),
        #                            nn.Dropout2d(0.1),
        #                            NonLinWeight(function=torch.sin, shape=14, batch_first=True, negative=False),
        #                            nn.AvgPool2d(kernel_size=2),
        #                            nn.Conv2d(64,16,kernel_size=(7,7), padding='same'),
        #                            nn.LeakyReLU(),
        #                            nn.BatchNorm2d(16),
        #                            nn.Dropout2d(0.2),
        #                            NonLinWeight(function=torch.cos, shape=7, batch_first=True, negative=True),
        #                            nn.AvgPool2d(kernel_size=2),
        #                            nn.Conv2d(16,64,kernel_size=(3,3), padding='same'),
        #                            nn.LeakyReLU(),
        #                            nn.BatchNorm2d(64),
        #                            nn.Dropout2d(0.1),
        #                            NonLinWeight(function=torch.sin, shape=3, batch_first=True, negative=False),
        #                            nn.AvgPool2d(kernel_size=3),
        #                            nn.Conv2d(64,64,kernel_size=(3,3), padding='same')
        #                            )
        # self.model = nn.Sequential(
        #                            weight_norm(nn.Conv2d(4,64,kernel_size=(3,3), padding='same'), dim=1),
        #                            nn.ReLU(),
        #                            nn.GroupNorm(2, 64),
        #                            nn.BatchNorm2d(64),
        #                            nn.Dropout2d(0.1),
        #                            NonLinWeight(function=torch.cos, shape=224, num_channels=64, batch_first=True, negative=False),
        #                            nn.AvgPool2d(kernel_size=2),
        #                            weight_norm(nn.Conv2d(64,256,kernel_size=(5,5), padding='same'), dim=1),
        #                            nn.LeakyReLU(),
        #                            nn.GroupNorm(32, 256),
        #                            nn.BatchNorm2d(256),
        #                            nn.Dropout2d(0.2),
        #                            NonLinWeight(function=torch.sin, shape=112, num_channels=256, batch_first=True, negative=True),
        #                            nn.AvgPool2d(kernel_size=2),
        #                            weight_norm(nn.Conv2d(256,128,kernel_size=(3,3), padding='same'), dim=1),
        #                            nn.LeakyReLU(),
        #                            nn.GroupNorm(16, 128),
        #                            nn.BatchNorm2d(128),
        #                            nn.Dropout2d(0.1),
        #                            NonLinWeight(function=torch.sin, shape=56, num_channels=128, batch_first=True, negative=False),
        #                            nn.AvgPool2d(kernel_size=2),
        #                            weight_norm(nn.Conv2d(128,32,kernel_size=(9,9), padding='same'), dim=1),
        #                            nn.LeakyReLU(),
        #                            nn.GroupNorm(4, 32),
        #                            nn.BatchNorm2d(32),
        #                            nn.Dropout2d(0.2),
        #                            NonLinWeight(function=torch.cos, shape=28, num_channels=32, batch_first=True, negative=True),
        #                            nn.AvgPool2d(kernel_size=2),
        #                            weight_norm(nn.Conv2d(32,64,kernel_size=(3,3), padding='same'), dim=1),
        #                            nn.LeakyReLU(),
        #                            nn.GroupNorm(8, 64),
        #                            nn.BatchNorm2d(64),
        #                            nn.Dropout2d(0.1),
        #                            NonLinWeight(function=torch.sin, shape=14, num_channels=64, batch_first=True, negative=False),
        #                            nn.AvgPool2d(kernel_size=2),
        #                            weight_norm(nn.Conv2d(64,128,kernel_size=(7,7), padding='same'), dim=1),
        #                            nn.LeakyReLU(),
        #                            nn.GroupNorm(32, 128),
        #                            nn.BatchNorm2d(128),
        #                            nn.Dropout2d(0.2),
        #                            NonLinWeight(function=torch.cos, shape=7, num_channels=128, batch_first=True, negative=True),
        #                            nn.AvgPool2d(kernel_size=2),
        #                            weight_norm(nn.Conv2d(128,32,kernel_size=(3,3), padding='same'), dim=1),
        #                            nn.LeakyReLU(),
        #                            nn.GroupNorm(2, 32),
        #                            nn.BatchNorm2d(32),
        #                            nn.Dropout2d(0.1),
        #                            NonLinWeight(function=torch.sin, shape=3, num_channels=32, batch_first=True, negative=False),
        #                            nn.AvgPool2d(kernel_size=3),
        #                            nn.Conv2d(32,64,kernel_size=(3,3), padding='same')
        #                            )
        
        self.model = nn.Sequential(nn.Conv2d(4,64,kernel_size=(3,3), padding='same'),
                                   nn.ReLU(),
                                   nn.BatchNorm2d(64),
                                   nn.Dropout2d(0.1),
                                   NonLinWeight(function=torch.cos, shape=224, num_channels=64, batch_first=True, negative=False),
                                   nn.AvgPool2d(kernel_size=2),
                                   nn.Conv2d(64,256,kernel_size=(5,5), padding='same'),
                                   nn.LeakyReLU(),
                                   nn.BatchNorm2d(256),
                                   nn.Dropout2d(0.2),
                                   NonLinWeight(function=torch.sin, shape=112, num_channels=256, batch_first=True, negative=True),
                                   nn.AvgPool2d(kernel_size=2),
                                   nn.Conv2d(256,128,kernel_size=(3,3), padding='same'),
                                   nn.LeakyReLU(),
                                   nn.BatchNorm2d(128),
                                   nn.Dropout2d(0.1),
                                   NonLinWeight(function=torch.sin, shape=56, num_channels=128, batch_first=True, negative=False),
                                   nn.AvgPool2d(kernel_size=2),
                                   nn.Conv2d(128,32,kernel_size=(9,9), padding='same'),
                                   nn.LeakyReLU(),
                                   nn.BatchNorm2d(32),
                                   nn.Dropout2d(0.2),
                                   NonLinWeight(function=torch.cos, shape=28, num_channels=32, batch_first=True, negative=True),
                                   nn.AvgPool2d(kernel_size=2),
                                   nn.Conv2d(32,64,kernel_size=(3,3), padding='same'),
                                #    nn.Conv2d(64,1,kernel_size=(3,3), padding='same'),
                                #    nn.AdaptiveAvgPool2d(output_size=(9,9))
                                   nn.LeakyReLU(),
                                   nn.BatchNorm2d(64),
                                   nn.Dropout2d(0.1),
                                   NonLinWeight(function=torch.sin, shape=14, num_channels=64, batch_first=True, negative=False),
                                   nn.AvgPool2d(kernel_size=2),
                                   nn.Conv2d(64,128,kernel_size=(7,7), padding='same'),
                                   nn.LeakyReLU(),
                                   nn.BatchNorm2d(128),
                                   nn.Dropout2d(0.2),
                                   NonLinWeight(function=torch.cos, shape=7, num_channels=128, batch_first=True, negative=True),
                                   nn.AvgPool2d(kernel_size=2),
                                   nn.Conv2d(128,32,kernel_size=(3,3), padding='same'),
                                   nn.LeakyReLU(),
                                   nn.BatchNorm2d(32),
                                   nn.Dropout2d(0.1),
                                   NonLinWeight(function=torch.sin, shape=3, num_channels=32, batch_first=True, negative=False),
                                   nn.AvgPool2d(kernel_size=3),
                                   nn.Conv2d(32,64,kernel_size=(3,3), padding='same')
                                   )
        

        # self.features_embedder = nn.Sequential(nn.Linear(163, 512),
        #                                        nn.Tanh(),
        #                                        nn.Dropout(0.1),
        #                                        nn.Linear(512, 128),
        #                                        nn.Tanh(),
        #                                        nn.Dropout(0.1),
        #                                        nn.Linear(128, 256),
        #                                        nn.Tanh(),
        #                                        nn.Linear(256, 64))
        # self.feat_kan = feat_kan = KAN(width=[163, 5,3, 32], grid=100, k=3, device='cuda')
        
        # self.features_embedder = nn.Sequential(
        #                                        feat_kan,
        #                                        )
        
        # self.cnn_classifier = nn.Sequential(nn.Flatten(start_dim=1, end_dim=-1),
        #                                     nn.Linear(64, 128),
        #                                     nn.LeakyReLU(),
        #                                     nn.Dropout(0.1),
        #                                     nn.Linear(128, 256),
        #                                     nn.LeakyReLU(),
        #                                     nn.Dropout(0.1),
        #                                     nn.Linear(256, 64),
        #                                     nn.LeakyReLU(),
        #                                     nn.Dropout(0.1),
        #                                     nn.Linear(64, 128))
        
        self.cnn_kan = cnn_kan = KAN(width=[64, 7, 13, 5, 9, 3, 6], grid=100, k=5, device='cuda')
        # self._fix_symoblics()
        self.cnn_classifier = nn.Sequential(nn.Flatten(start_dim=1, end_dim=-1),
                                            cnn_kan
                                            )
        self.once = True
        # self.feat_cnn_classifier = nn.Bilinear(64, 32, 64)
        
        # self.classifier = nn.Sequential(nn.Dropout(0.1),
        #                                 nn.Linear(128, 256),
        #                                 nn.ReLU(),
        #                                 nn.Dropout(0.1),
        #                                 nn.Linear(256, 128),
        #                                 nn.ReLU(),
        #                                 nn.Linear(128, 6))
        
        # self.class_kan = class_kan = KAN(width=[64, 7, 3, 6], grid=100, k=3, device='cuda')
        
        # self.classifier = nn.Sequential(
        #                                 class_kan
        #                                 )
    
    def _fix_symoblics(self):
        # (layer idx, input neuron idx, output neuron idx)
        
        # ---------------- Layer 0 ------------------
        # 7 neurons
        self.cnn_kan.fix_symbolic(0, 0, 0, 'x^2')
        self.cnn_kan.fix_symbolic(0, 1, 0, 'x^2')
        self.cnn_kan.fix_symbolic(0, 2, 0, 'x^2')
        
        self.cnn_kan.fix_symbolic(0, 0, 1, 'x^3')
        self.cnn_kan.fix_symbolic(0, 1, 1, 'x^3')
        self.cnn_kan.fix_symbolic(0, 2, 1, 'x^3')
        
        # self.cnn_kan.fix_symbolic(0, 4, 2, 'log')
        
        # self.cnn_kan.fix_symbolic(0, 6, 3, 'cosh')
        
        # self.cnn_kan.fix_symbolic(0, 8, 4, 'tanh')
        
        self.cnn_kan.fix_symbolic(0, 10, 5, '1/x^2')
        self.cnn_kan.fix_symbolic(0, 11, 5, '1/x^2')
        self.cnn_kan.fix_symbolic(0, 12, 5, '1/x^2')
        
        self.cnn_kan.fix_symbolic(0, 10, 6, '1/x^3')
        self.cnn_kan.fix_symbolic(0, 11, 6, '1/x^3')
        self.cnn_kan.fix_symbolic(0, 12, 6, '1/x^3')
        
        # ---------------- Layer 1 ------------------
        # 13 neurons
        
        # Chunk one
        # self.cnn_kan.fix_symbolic(1, 0, 0, 'x')
        self.cnn_kan.fix_symbolic(1, 1, 0, 'gaussian')
        
        self.cnn_kan.fix_symbolic(1, 0, 1, 'tanh')
        # self.cnn_kan.fix_symbolic(1, 1, 1, 'x')
        
        # self.cnn_kan.fix_symbolic(1, 0, 2, 'x')
        self.cnn_kan.fix_symbolic(1, 1, 2, 'sin')
        
        # Chunk two
        # self.cnn_kan.fix_symbolic(1, 0, 3, 'x')
        self.cnn_kan.fix_symbolic(1, 1, 3, 'cosh')
        
        # self.cnn_kan.fix_symbolic(1, 0, 4, 'x')
        # self.cnn_kan.fix_symbolic(1, 1, 4, 'x')
        
        # self.cnn_kan.fix_symbolic(1, 0, 5, 'x')
        # self.cnn_kan.fix_symbolic(1, 1, 5, 'exp')
        
        # Chunk three
        # self.cnn_kan.fix_symbolic(1, 1, 6, 'x')
        self.cnn_kan.fix_symbolic(1, 2, 6, 'cosh')
        # self.cnn_kan.fix_symbolic(1, 3, 6, 'x')
        
        # Chunk four
        self.cnn_kan.fix_symbolic(1, 3, 7, 'sigmoid')
        # self.cnn_kan.fix_symbolic(1, 4, 7, 'x')
        
        self.cnn_kan.fix_symbolic(1, 3, 8, '1/x')
        # self.cnn_kan.fix_symbolic(1, 4, 8, 'x')
        
        # self.cnn_kan.fix_symbolic(1, 3, 9, 'x')
        self.cnn_kan.fix_symbolic(1, 4, 9, 'x^4')
        
        # Chunk five
        self.cnn_kan.fix_symbolic(1, 3, 10, '1/sqrt(x)')
        # self.cnn_kan.fix_symbolic(1, 4, 10, 'x')
        
        # self.cnn_kan.fix_symbolic(1, 3, 11, 'x')
        # self.cnn_kan.fix_symbolic(1, 4, 11, 'log')
        
        # self.cnn_kan.fix_symbolic(1, 3, 12, 'x')
        self.cnn_kan.fix_symbolic(1, 4, 12, '0')

        # ---------------- Layer 2 ------------------
        # 5 neurons
        
        # Chunk 1
        self.cnn_kan.fix_symbolic(2, 0, 0, '1/x^3')
        self.cnn_kan.fix_symbolic(2, 1, 0, 'sin')
        # self.cnn_kan.fix_symbolic(2, 2, 0, 'exp')
        
        self.cnn_kan.fix_symbolic(2, 0, 1, '1/x^4')
        self.cnn_kan.fix_symbolic(2, 1, 1, 'sin')
        # self.cnn_kan.fix_symbolic(2, 2, 1, 'sgn')
        
        # Chunk 2
        self.cnn_kan.fix_symbolic(2, 3, 2, 'sqrt')
        # self.cnn_kan.fix_symbolic(2, 4, 2, 'tan')
        self.cnn_kan.fix_symbolic(2, 5, 2, 'x^2')
        
        # Chunk 3
        # self.cnn_kan.fix_symbolic(2, 6, 3, 'gaussian')
        # self.cnn_kan.fix_symbolic(2, 7, 3, 'cosh')
        self.cnn_kan.fix_symbolic(2, 8, 3, 'x^4')
        
        # self.cnn_kan.fix_symbolic(2, 6, 4, 'sgn')
        # self.cnn_kan.fix_symbolic(2, 7, 4, 'cosh')
        self.cnn_kan.fix_symbolic(2, 8, 4, '1/x^2')
        
        # ---------------- Layer 3 ------------------
        # 9 neurons
        
        # Chunk 1
        # self.cnn_kan.fix_symbolic(3, 0, 0, 'x^4')
        # self.cnn_kan.fix_symbolic(3, 0, 1, 'tan')
        # self.cnn_kan.fix_symbolic(3, 0, 2, 'exp')
        
        # self.cnn_kan.fix_symbolic(3, 1, 0, 'x')
        self.cnn_kan.fix_symbolic(3, 1, 1, 'sqrt')
        # self.cnn_kan.fix_symbolic(3, 1, 2, 'x')
        
        self.cnn_kan.fix_symbolic(3, 2, 0, '1/x^2')
        # self.cnn_kan.fix_symbolic(3, 2, 1, 'log')
        # self.cnn_kan.fix_symbolic(3, 2, 2, 'x')
        
        # Chunk 2
        self.cnn_kan.fix_symbolic(3, 0, 3, 'x^3')
        self.cnn_kan.fix_symbolic(3, 0, 4, 'sin')
        # self.cnn_kan.fix_symbolic(3, 0, 5, 'log')
        
        # self.cnn_kan.fix_symbolic(3, 1, 3, 'x')
        self.cnn_kan.fix_symbolic(3, 1, 4, 'cosh')
        # self.cnn_kan.fix_symbolic(3, 1, 5, 'x')
        
        self.cnn_kan.fix_symbolic(3, 2, 3, '1/x^3')
        self.cnn_kan.fix_symbolic(3, 2, 4, 'arctan')
        # self.cnn_kan.fix_symbolic(3, 2, 5, 'x')
        
        # Chunk 3
        self.cnn_kan.fix_symbolic(3, 0, 6, 'x^2')
        self.cnn_kan.fix_symbolic(3, 0, 7, 'cosh')
        # self.cnn_kan.fix_symbolic(3, 0, 8, 'abs')
        
        # self.cnn_kan.fix_symbolic(3, 2, 6, 'x')
        self.cnn_kan.fix_symbolic(3, 2, 7, '1/sqrt(x)')
        # self.cnn_kan.fix_symbolic(3, 2, 8, 'x')
        
        self.cnn_kan.fix_symbolic(3, 1, 6, '1/x^2')
        self.cnn_kan.fix_symbolic(3, 1, 7, 'arctanh')
        # self.cnn_kan.fix_symbolic(3, 1, 8, 'x')
        
        # ---------------- Layer 3 ------------------
        # 3 neurons
        # out to 6
        
        # Fully connected
        
        self.cnn_kan.fix_symbolic(4, 0, 0, 'tanh')
        self.cnn_kan.fix_symbolic(4, 1, 0, 'tanh')
        self.cnn_kan.fix_symbolic(4, 2, 0, 'tanh')
        self.cnn_kan.fix_symbolic(4, 3, 0, 'tanh')
        self.cnn_kan.fix_symbolic(4, 4, 0, 'tanh')
        self.cnn_kan.fix_symbolic(4, 5, 0, 'tanh')
        
        self.cnn_kan.fix_symbolic(4, 0, 1, 'tanh')
        self.cnn_kan.fix_symbolic(4, 1, 1, 'tanh')
        self.cnn_kan.fix_symbolic(4, 2, 1, 'tanh')
        self.cnn_kan.fix_symbolic(4, 3, 1, 'tanh')
        self.cnn_kan.fix_symbolic(4, 4, 1, 'tanh')
        self.cnn_kan.fix_symbolic(4, 5, 1, 'tanh')
        
        self.cnn_kan.fix_symbolic(4, 0, 2, 'tanh')
        self.cnn_kan.fix_symbolic(4, 1, 2, 'tanh')
        self.cnn_kan.fix_symbolic(4, 2, 2, 'tanh')
        self.cnn_kan.fix_symbolic(4, 3, 2, 'tanh')
        self.cnn_kan.fix_symbolic(4, 4, 2, 'tanh')
        self.cnn_kan.fix_symbolic(4, 5, 2, 'tanh')
        
        
    def forward(self, input, features):
        # img = self.cnn_classifier(self.model(input))
        img = self.model(input)
        # print(img.shape)
        img_class = self.cnn_classifier(img)
        # if self.once:
        #     self._fix_symoblics()
        #     self.once = False
        # feat = self.features_embedder(features)
        # blin = self.feat_cnn_classifier(img_class, feat)
        # out = self.classifier(img_class)
        return img_class

# 0.4 Loss model arch for NLWCNN
        
        # self.model = nn.Sequential(nn.Conv2d(4,64,kernel_size=(3,3), padding='same'),
        #                            nn.ReLU(),
        #                            nn.BatchNorm2d(64),
        #                            nn.Dropout2d(0.1),
        #                            NonLinWeight(function=torch.cos, shape=224, num_channels=64, batch_first=True, negative=False),
        #                            nn.AvgPool2d(kernel_size=2),
        #                            nn.Conv2d(64,256,kernel_size=(5,5), padding='same'),
        #                            nn.LeakyReLU(),
        #                            nn.BatchNorm2d(256),
        #                            nn.Dropout2d(0.2),
        #                            NonLinWeight(function=torch.sin, shape=112, num_channels=256, batch_first=True, negative=True),
        #                            nn.AvgPool2d(kernel_size=2),
        #                            nn.Conv2d(256,128,kernel_size=(3,3), padding='same'),
        #                            nn.LeakyReLU(),
        #                            nn.BatchNorm2d(128),
        #                            nn.Dropout2d(0.1),
        #                            NonLinWeight(function=torch.sin, shape=56, num_channels=128, batch_first=True, negative=False),
        #                            nn.AvgPool2d(kernel_size=2),
        #                            nn.Conv2d(128,32,kernel_size=(9,9), padding='same'),
        #                            nn.LeakyReLU(),
        #                            nn.BatchNorm2d(32),
        #                            nn.Dropout2d(0.2),
        #                            NonLinWeight(function=torch.cos, shape=28, num_channels=32, batch_first=True, negative=True),
        #                            nn.AvgPool2d(kernel_size=2),
        #                            nn.Conv2d(32,64,kernel_size=(3,3), padding='same'),
        #                            nn.LeakyReLU(),
        #                            nn.BatchNorm2d(64),
        #                            nn.Dropout2d(0.1),
        #                            NonLinWeight(function=torch.sin, shape=14, num_channels=64, batch_first=True, negative=False),
        #                            nn.AvgPool2d(kernel_size=2),
        #                            nn.Conv2d(64,128,kernel_size=(7,7), padding='same'),
        #                            nn.LeakyReLU(),
        #                            nn.BatchNorm2d(128),
        #                            nn.Dropout2d(0.2),
        #                            NonLinWeight(function=torch.cos, shape=7, num_channels=128, batch_first=True, negative=True),
        #                            nn.AvgPool2d(kernel_size=2),
        #                            nn.Conv2d(128,32,kernel_size=(3,3), padding='same'),
        #                            nn.LeakyReLU(),
        #                            nn.BatchNorm2d(32),
        #                            nn.Dropout2d(0.1),
        #                            NonLinWeight(function=torch.sin, shape=3, num_channels=32, batch_first=True, negative=False),
        #                            nn.AvgPool2d(kernel_size=3),
        #                            nn.Conv2d(32,64,kernel_size=(3,3), padding='same')
        #                            )
        # self.features_embedder = nn.Sequential(nn.Linear(163, 512),
        #                                        nn.Tanh(),
        #                                        nn.Dropout(0.1),
        #                                        nn.Linear(512, 128),
        #                                        nn.Tanh(),
        #                                        nn.Dropout(0.1),
        #                                        nn.Linear(128, 256),
        #                                        nn.Tanh(),
        #                                        nn.Linear(256, 64))
        
        # self.cnn_classifier = nn.Sequential(nn.Flatten(start_dim=1, end_dim=-1),
        #                                     nn.Linear(64, 128),
        #                                     nn.LeakyReLU(),
        #                                     nn.Dropout(0.1),
        #                                     nn.Linear(128, 256),
        #                                     nn.LeakyReLU(),
        #                                     nn.Dropout(0.1),
        #                                     nn.Linear(256, 64),
        #                                     nn.LeakyReLU(),
        #                                     nn.Dropout(0.1),
        #                                     nn.Linear(64, 128))
        
        # self.feat_cnn_classifier = nn.Bilinear(128, 64, 128)
        
        # self.classifier = nn.Sequential(nn.Dropout(0.1),
        #                                 nn.Linear(128, 256),
        #                                 nn.ReLU(),
        #                                 nn.Dropout(0.1),
        #                                 nn.Linear(256, 128),
        #                                 nn.ReLU(),
        #                                 nn.Linear(128, 6))
        
        
class BestModel(nn.Module):
    def __init__(self, ):
        super().__init__()
        from NonLinWeight import NonLinWeight
        from torch.nn.utils.parametrizations import weight_norm
        
        # 0.4 Loss model arch for NLWCNN
        
        self.model = nn.Sequential(nn.Conv2d(4,64,kernel_size=(3,3), padding='same'),
                                   nn.ReLU(),
                                   nn.BatchNorm2d(64),
                                   nn.Dropout2d(0.1),
                                   NonLinWeight(function=torch.cos, shape=224, num_channels=64, batch_first=True, negative=False),
                                   nn.AvgPool2d(kernel_size=2),
                                   nn.Conv2d(64,256,kernel_size=(5,5), padding='same'),
                                   nn.LeakyReLU(),
                                   nn.BatchNorm2d(256),
                                   nn.Dropout2d(0.2),
                                   NonLinWeight(function=torch.sin, shape=112, num_channels=256, batch_first=True, negative=True),
                                   nn.AvgPool2d(kernel_size=2),
                                   nn.Conv2d(256,128,kernel_size=(3,3), padding='same'),
                                   nn.LeakyReLU(),
                                   nn.BatchNorm2d(128),
                                   nn.Dropout2d(0.1),
                                   NonLinWeight(function=torch.sin, shape=56, num_channels=128, batch_first=True, negative=False),
                                   nn.AvgPool2d(kernel_size=2),
                                   nn.Conv2d(128,32,kernel_size=(9,9), padding='same'),
                                   nn.LeakyReLU(),
                                   nn.BatchNorm2d(32),
                                   nn.Dropout2d(0.2),
                                   NonLinWeight(function=torch.cos, shape=28, num_channels=32, batch_first=True, negative=True),
                                   nn.AvgPool2d(kernel_size=2),
                                   nn.Conv2d(32,64,kernel_size=(3,3), padding='same'),
                                   nn.LeakyReLU(),
                                   nn.BatchNorm2d(64),
                                   nn.Dropout2d(0.1),
                                   NonLinWeight(function=torch.sin, shape=14, num_channels=64, batch_first=True, negative=False),
                                   nn.AvgPool2d(kernel_size=2),
                                   nn.Conv2d(64,128,kernel_size=(7,7), padding='same'),
                                   nn.LeakyReLU(),
                                   nn.BatchNorm2d(128),
                                   nn.Dropout2d(0.2),
                                   NonLinWeight(function=torch.cos, shape=7, num_channels=128, batch_first=True, negative=True),
                                   nn.AvgPool2d(kernel_size=2),
                                   nn.Conv2d(128,32,kernel_size=(3,3), padding='same'),
                                   nn.LeakyReLU(),
                                   nn.BatchNorm2d(32),
                                   nn.Dropout2d(0.1),
                                   NonLinWeight(function=torch.sin, shape=3, num_channels=32, batch_first=True, negative=False),
                                   nn.AvgPool2d(kernel_size=3),
                                   nn.Conv2d(32,64,kernel_size=(3,3), padding='same')
                                   )
        self.features_embedder = nn.Sequential(nn.Linear(163, 512),
                                               nn.Tanh(),
                                               nn.Dropout(0.1),
                                               nn.Linear(512, 128),
                                               nn.Tanh(),
                                               nn.Dropout(0.1),
                                               nn.Linear(128, 256),
                                               nn.Tanh(),
                                               nn.Linear(256, 64))
        
        self.cnn_classifier = nn.Sequential(nn.Flatten(start_dim=1, end_dim=-1),
                                            nn.Linear(64, 128),
                                            nn.LeakyReLU(),
                                            nn.Dropout(0.1),
                                            nn.Linear(128, 256),
                                            nn.LeakyReLU(),
                                            nn.Dropout(0.1),
                                            nn.Linear(256, 64),
                                            nn.LeakyReLU(),
                                            nn.Dropout(0.1),
                                            nn.Linear(64, 128))
        
        self.feat_cnn_classifier = nn.Bilinear(128, 64, 128)
        
        self.classifier = nn.Sequential(nn.Dropout(0.1),
                                        nn.Linear(128, 256),
                                        nn.ReLU(),
                                        nn.Dropout(0.1),
                                        nn.Linear(256, 128),
                                        nn.ReLU(),
                                        nn.Linear(128, 6))
        
    def forward(self, img, feat):
        img = self.cnn_classifier(self.model(img))
        feat = self.features_embedder(feat)
        
        conn = self.feat_cnn_classifier(img, feat)
        
        out = self.classifier(conn)
        
        return out