import torch # All the torch modules
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
#import torchvision.models as models

#-------------------------------------------------------

# Important code for Knowledge Transfer anurag kumar model
import weak_feature_extractor.network_architectures as netark
from collections import OrderedDict
#pre_model_path = 'weak_feature_extractor/mx-h64-1024_0d3-1.17.pkl'
netType = getattr(netark, 'weak_mxh64_1024')
netwrkgpl = F.max_pool2d # keep it fixed. It was avg_pool2d for experiments on ESC, initial experiments on SONYC_UST

class FtEx(nn.Module):
    def __init__(self, n_classes=10, multi_label=False):
        super(FtEx, self).__init__()
        self.netx = netType(527, netwrkgpl) # Only initially to load model
        #self.load_model(pre_model_path)
        self.layer = nn.Sequential(nn.Conv2d(1024, 256, kernel_size=1), nn.ReLU())
        self.fc = nn.Linear(256, n_classes, bias=True)
        self.reg = nn.Dropout(0.2)

    def forward(self, inp):
        out, out_inter = self.netx(inp)
        out = self.layer(out_inter[0])
        out = torch.flatten( netwrkgpl(out, kernel_size=out.shape[2:]), 1)
        out = self.reg(out)
        out = self.fc(out)
        return out, out_inter

    def load_model(self, modpath):
        #load through cpu -- safest
        state_dict = torch.load(modpath,map_location=lambda storage, loc: storage)
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if 'module.' in k:
                name = k[7:]
            else:
                name = k
            new_state_dict[name] = v
        self.netx.load_state_dict(new_state_dict)


class FtEx_v2(nn.Module):
    def __init__(self, n_classes=10, mid_size=256):
        super(FtEx_v2, self).__init__()
        self.netx = netType(527, netwrkgpl) # Only initially to load model
        #self.load_model(pre_model_path)
        self.layer = nn.Sequential(nn.Conv2d(1024, mid_size, kernel_size=1), nn.ReLU())
        self.fc = nn.Linear(mid_size, n_classes, bias=True)
        self.reg = nn.Dropout(0.15)

    def forward(self, inp):
        out, out_inter = self.netx(inp)
        out = self.layer(out_inter[0])
        out = torch.flatten( netwrkgpl(out, kernel_size=out.shape[2:]), 1)
        out = self.reg(out)
        out = self.fc(out)
        return out, out_inter

    def load_model(self, modpath):
        #load through cpu -- safest
        state_dict = torch.load(modpath,map_location=lambda storage, loc: storage)
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if 'module.' in k:
                name = k[7:]
            else:
                name = k
            new_state_dict[name] = v
        self.netx.load_state_dict(new_state_dict)




class HNet_FtEx(nn.Module):
    def __init__(self, N_COMP=30, T=227, in_maps=256):
        super(HNet_FtEx, self).__init__()
        self.upsamp = nn.UpsamplingBilinear2d(scale_factor=(1, 2))
        self.upsamp2 = nn.UpsamplingBilinear2d(size=[1, T])
        self.conv1 = nn.Conv2d(in_maps, 256, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        #self.conv3 = nn.Conv2d(128, 128, kernel_size=5, padding=2)

        self.pool = nn.AvgPool2d(kernel_size=(2, 1))
        self.conv4 = nn.Conv1d(256, N_COMP, kernel_size=5, padding=2)
        self.conv5 = nn.Conv1d(N_COMP, N_COMP, kernel_size=5, padding=2)
        self.activ = nn.ReLU()

        self.conv6 = nn.Conv2d(256, 256, kernel_size=(4, 1), padding=0)

        self.conv7 = nn.Conv2d(2*in_maps, 256, kernel_size=3, padding=1)

    def forward(self, inp):
        # Assume input of shape n_batch x 256 x 30 x 27
        x = self.upsamp(inp[2].transpose(2, 3))
        x2 = self.upsamp(inp[1].transpose(2, 3))
        #print (x.shape, x2.shape)
        x = self.pool( self.activ( self.conv1(x) ) )
        x2 = self.upsamp( self.activ( self.conv7(x2) ) )
        #print (x.shape, x2.shape)
        #x = torch.cat((x, x), dim=1) # FOr single hidden layer exp
        x = torch.cat((x, x2), dim=1)
        x = self.upsamp(x)
        #x = self.pool( self.pool( self.activ( self.conv2(x) ) ) )
        x = self.activ( self.conv2(x) )
        x = self.activ( self.conv6(x) )
        x = self.upsamp2(x)
        x = torch.flatten(x, 2)
        x = self.activ( self.conv4(x) )
        output = self.activ( self.conv5(x) )
       
        
        return output 


class HNet_FtEx_v2(nn.Module):
    def __init__(self, N_COMP=30, T=227, in_maps=256):
        super(HNet_FtEx_v2, self).__init__()
        self.upsamp = nn.UpsamplingBilinear2d(scale_factor=(1, 2))
        self.upsamp2 = nn.UpsamplingBilinear2d(size=[1, T])
        self.conv1 = nn.Conv2d(in_maps, 256, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        #self.conv3 = nn.Conv2d(128, 128, kernel_size=5, padding=2)

        self.pool = nn.AvgPool2d(kernel_size=(2, 1))
        self.conv4 = nn.Conv1d(256, N_COMP, kernel_size=5, padding=2)
        self.conv5 = nn.Conv1d(N_COMP, N_COMP, kernel_size=5, padding=2)
        self.activ = nn.ReLU()

        self.conv6 = nn.Conv2d(256, 256, kernel_size=(4, 1), padding=0)

        self.conv7 = nn.Conv2d(2*in_maps, 128, kernel_size=3, padding=1)
        self.conv8 = nn.Conv2d(128, 128, kernel_size=(4, 1), stride=(4, 1), padding=0)

    def forward(self, inp):
        # Assume input of shape n_batch x 256 x 30 x 27
        #print (inp[1].shape, inp[2].shape, inp[3].shape)
        x2 = self.upsamp(inp[2].transpose(2, 3))
        x1 = self.upsamp(inp[1].transpose(2, 3))
        x3 = inp[3].transpose(2, 3)[:, :, :, :52]
        #print (x1.shape, x2.shape, x3.shape)
        x2 = self.pool( self.activ( self.conv1(x2) ) )
        x1 = self.upsamp( self.activ( self.conv7(x1) ) )
        x3 = self.activ( self.conv8(x3) )
        #print (x1.shape, x2.shape, x3.shape)
        x = torch.cat((x2, x1, x3), dim=1)
        x = self.upsamp(x)
        x = self.upsamp( self.activ( self.conv2(x) ) )
        x = self.activ( self.conv6(x) )
        x = self.upsamp2(x)
        x = torch.flatten(x, 2)
        x = self.activ( self.conv4(x) )
        output = self.activ( self.conv5(x) )
       
        
        return output 


class HNet_FtEx_general(nn.Module):
    def __init__(self, N_COMP=30, T=227, in_maps=256):
        super(HNet_FtEx_general, self).__init__()
        self.upsamp = nn.UpsamplingBilinear2d(scale_factor=(1, 2))
        self.upsamp2 = nn.UpsamplingBilinear2d(size=[1, T])
        self.conv1 = nn.Conv2d(in_maps, 256, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        #self.conv3 = nn.Conv2d(128, 128, kernel_size=5, padding=2)

        self.pool = nn.AvgPool2d(kernel_size=(2, 1))
        self.conv4 = nn.Conv1d(256, N_COMP, kernel_size=5, padding=2)
        self.conv5 = nn.Conv1d(N_COMP, N_COMP, kernel_size=5, padding=2)
        self.activ = nn.ReLU()

        self.conv6 = nn.Conv2d(256, 256, kernel_size=(4, 1), padding=0)

        self.conv7 = nn.Conv2d(2*in_maps, 128, kernel_size=3, padding=1)
        self.conv8 = nn.Conv2d(128, 128, kernel_size=(4, 1), stride=(4, 1), padding=0)

    def forward(self, inp):
        # Assume 3 inputs of shape n_batch x 256 x 30 x 27
        #print (inp[1].shape, inp[2].shape, inp[3].shape)
        maps_channel_T = int(inp[1].shape[2])
        x2 = self.upsamp(inp[2].transpose(2, 3)[:, :, :, :2*maps_channel_T])
        x1 = self.upsamp(inp[1].transpose(2, 3))
        x3 = inp[3].transpose(2, 3)[:, :, :, :4*maps_channel_T]
        #print (x1.shape, x2.shape, x3.shape)
        x2 = self.pool( self.activ( self.conv1(x2) ) )
        x1 = self.upsamp( self.activ( self.conv7(x1) ) )
        x3 = self.activ( self.conv8(x3) )
        #print (x1.shape, x2.shape, x3.shape)
        x = torch.cat((x2, x1, x3), dim=1)
        x = self.upsamp(x)
        x = self.upsamp( self.activ( self.conv2(x) ) )
        x = self.activ( self.conv6(x) )
        x = self.upsamp2(x)
        x = torch.flatten(x, 2)
        x = self.activ( self.conv4(x) )
        output = self.activ( self.conv5(x) )
       
        
        return output 





class explainer(nn.Module):
    def __init__(self, N_COMP=30, N_FRAMES=20, n_classes=10, dropout=False):
        super(explainer, self).__init__()
        self.fc1 = nn.Linear(N_COMP*N_FRAMES, n_classes, bias=False)
        self.pool = nn.AdaptiveMaxPool1d(N_FRAMES)
        if dropout:
            self.drop = nn.Dropout(0.3)
        else:
            self.drop = nn.Dropout(0.000000001) 

    def forward(self, inp):
        x = self.drop( self.pool(inp) )
        x = torch.flatten(x, 1)
        output = self.fc1(x)
        return output


class explainer_v2(nn.Module):
    def __init__(self, N_COMP=30, N_FRAMES=20, n_classes=10, dropout=False):
        super(explainer_v2, self).__init__()
        self.fc1 = nn.Linear(N_COMP*N_FRAMES, 256, bias=False)
        self.fc2 = nn.Linear(256, n_classes, bias=False)
        self.pool = nn.AdaptiveMaxPool1d(N_FRAMES)
        self.activ = nn.ReLU()
        if dropout:
            self.drop = nn.Dropout(0.3)
        else:
            self.drop = nn.Dropout(0.000000001) 

    def forward(self, inp):
        x = self.drop( self.pool(inp) )
        x = torch.flatten(x, 1)
        x = self.activ( self.fc1(x) )
        output = self.fc2(x)
        return output


class explainer_v3(nn.Module):
    def __init__(self, N_COMP=30, n_classes=10, dropout=False):
        super(explainer_v3, self).__init__()
        self.M = N_COMP
        self.L = 256
        self.cdim = 1 # Coefficient/weight dimension for attention, just a single real number
        self.attention = nn.Sequential( nn.Linear(self.M, self.L),
                                        nn.Tanh(),
                                        nn.Linear(self.L, self.cdim)
                                      )
        self.fc1 = nn.Linear(self.M, n_classes, bias=False)
        if dropout:
            self.drop = nn.Dropout(0.3)
        else:
            self.drop = nn.Dropout(0.000000001) 

    def forward(self, inp):
        # inp shape if [N_BATCH x N_COMP x TIME_FRAMES]
        A = self.attention(torch.transpose(self.drop(inp), 1, 2))
        # A should be of shape [N_BATCH x TIME_FRAMES x 1]
        A = nn.Softmax(dim=1)(A)
        x = torch.bmm(inp, A)[:, :, 0]
        #print (x.shape)
        output = self.fc1(x)
        return output
	
        

class explainer_v4(nn.Module):
    def __init__(self, N_COMP=30, n_classes=10, dropout=False):
        super(explainer_v4, self).__init__()
        self.M = N_COMP
        self.L = 256
        self.cdim = 1 # Coefficient/weight dimension for attention, just a single real number
        self.attention_V = nn.Sequential(
            nn.Linear(self.M, self.L),
            nn.Tanh()
        )

        self.attention_U = nn.Sequential(
            nn.Linear(self.M, self.L),
            nn.ReLU()
        )

        self.attention_weights = nn.Linear(self.L, self.cdim)
                                   
        self.fc1 = nn.Linear(self.M, n_classes, bias=False)
        if dropout:
            self.drop = nn.Dropout(0.4)
        else:
            self.drop = nn.Dropout(0.000000001) 

    def forward(self, inp):
        # inp shape if [N_BATCH x N_COMP x TIME_FRAMES]
        A_V = self.attention_V(torch.transpose(self.drop(inp), 1, 2))
        A_U = self.attention_U(torch.transpose(self.drop(inp), 1, 2))
        # A should be of shape [N_BATCH x TIME_FRAMES x 1]
        A = self.attention_weights(A_V * A_U)
        A = nn.Softmax(dim=1)(A)
        x = torch.bmm(inp, A)[:, :, 0]
        #print (x.shape)
        output = self.fc1(x)
        return output

    def return_pooled_activ(self, inp):
        #n_samp = int(inp.shape[0])
        #assert len(class_idx) == n_samp, "Class index should be list or array of same length as number of samples"
        A_V = self.attention_V(torch.transpose(inp, 1, 2))
        A_U = self.attention_U(torch.transpose(inp, 1, 2))
        A = self.attention_weights(A_V * A_U)
        A = nn.Softmax(dim=1)(A)
        x = torch.bmm(inp, A)[:, :, 0]
        #weights = self.fc1.weight
        #for i in range(n_samp):
        #    x[i] = x[i] * weights[int(class_idx[i])]
        return x



class NMF_D(nn.Module):
    def __init__(self,N_COMP=30, FREQ=513, init_file=None):
        super(NMF_D, self).__init__()
        self.W = nn.Parameter( torch.rand(FREQ, N_COMP), requires_grad=True )
        self.activ = nn.ReLU()
        if init_file is not None:
            init_W = torch.as_tensor(np.load(init_file)).float()
            self.W = nn.Parameter( init_W, requires_grad=True )

    def forward(self, inp):
        # Assume input of shape n_batch x n_comp x T
        W = self.activ(self.W)
        W = nn.functional.normalize(W, dim=0, p=2)
        W = torch.stack(int(inp.shape[0]) * [W])
        output = self.activ( torch.bmm(W, inp) ) 
        return output

    def return_W(self, dtype='numpy'):
        W = self.W
        W = nn.functional.normalize(self.activ(W), dim=0, p=2)
        if dtype == 'numpy':
            return W.cpu().data.numpy()
        else:
            return W



class SB_CNN(nn.Module):
    def __init__(self, n_classes=10, in_maps=1):
        super(SB_CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_maps, out_channels=24, kernel_size=5, padding=2)
        self.maxpool = nn.MaxPool2d((2, 4))
        self.pool = nn.AdaptiveAvgPool2d((10, 10))
        self.conv2 = nn.Conv2d(in_channels=24, out_channels=48, kernel_size=5, padding=2)
        self.activ = nn.ReLU()
        self.conv3 = nn.Conv2d(in_channels=48, out_channels=48, kernel_size=5, padding=2)
        self.fc1 = nn.Linear(48*10*10, 128, bias=True)
        self.fc2 = nn.Linear(128, 10, bias=True)

    def forward(self, x):
        out = self.maxpool( self.conv1(x) )
        out = self.activ(out)
        out = self.maxpool( self.conv2(out) )
        out = self.activ(out)
        out_inter = self.activ( self.conv3(out) )
        out = self.pool(out_inter)
        out = torch.flatten(out, 1)
        out = self.activ( self.fc1(out) )
        out = self.fc2(out)
        return out, out_inter








#########---------------------------- NETWORKS FOR BASELINE FLINT -------------------------- ###########



class baseline_FLINT_h(nn.Module):
    def __init__(self, in_size=30, n_classes=8):
        super(baseline_FLINT_h, self).__init__()
        self.fc1 = nn.Linear(in_size, n_classes, bias=False)
        self.drop = nn.Dropout(0.01)

    def forward(self, inp):
        x = self.drop(inp)
        return self.fc1(x)



class decoder_SUST(nn.Module):
    def __init__(self, in_size=80, T=862, F=513):
        super(decoder_SUST, self).__init__()
        self.fc1 = nn.Linear(in_size, 16*100, bias=True)
        self.conv1 = nn.Conv1d(16, 32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.conv4 = nn.Conv1d(128, F, kernel_size=3, padding=1)
        self.activ = nn.ReLU()
        self.upsamp2 = nn.UpsamplingBilinear2d(size=[128, T])

    def forward(self, inp):
        x = self.fc1(inp)
        x = x.view(-1, 16, 100)
        x = self.activ( self.conv1(x) )
        x = F.interpolate(x, scale_factor=2)
        x = self.activ( self.conv2(x) )
        x = F.interpolate(x, scale_factor=2)
        x = self.activ( self.conv3(x) )
        x = self.upsamp2(x.unsqueeze(1))[:, 0, :, :]
        x = self.activ( self.conv4(x) )
        #x = x.transpose(1, 2)
        return x


class Psi(nn.Module):
    def __init__(self, out_size=80, in_maps=256):
        super(Psi, self).__init__()
        self.upsamp = nn.UpsamplingBilinear2d(scale_factor=(1, 2))
        #self.upsamp2 = nn.UpsamplingBilinear2d(size=[1, T])
        self.conv1 = nn.Conv2d(in_maps, 256, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(512, 128, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(3)
        self.pool2 = nn.AvgPool2d(kernel_size=(2, 1))
        self.activ = nn.ReLU()

        self.conv7 = nn.Conv2d(2*in_maps, 128, kernel_size=3, padding=1)
        self.conv8 = nn.Conv2d(128, 128, kernel_size=(4, 1), stride=(4, 1), padding=0)
        self.fc1 = nn.Linear(128*3*3, out_size)

    def forward(self, inp):
        # Assume 3 inputs of shape n_batch x 256 x 30 x 27
        #print (inp[1].shape, inp[2].shape, inp[3].shape)
        maps_channel_T = int(inp[1].shape[2])
        x2 = self.upsamp(inp[2].transpose(2, 3)[:, :, :, :2*maps_channel_T])
        x1 = self.upsamp(inp[1].transpose(2, 3))
        x3 = inp[3].transpose(2, 3)[:, :, :, :4*maps_channel_T]
        #print (x1.shape, x2.shape, x3.shape)
        x2 = self.pool2( self.activ( self.conv1(x2) ) )
        x1 = self.upsamp( self.activ( self.conv7(x1) ) )
        x3 = self.activ( self.conv8(x3) )
        #print (x1.shape, x2.shape, x3.shape)
        x = torch.cat((x2, x1, x3), dim=1)
        x = self.activ( self.conv2(x)  )       
        x = self.pool(x).view(-1, 128*3*3)
        output = self.activ( self.fc1(x) )
        return output 



# --------------------------  Experimental things ----------------------- #

class HNet_FtEx_stupid(nn.Module):
    def __init__(self, N_COMP=30, T=227, in_maps=1):
        super(HNet_FtEx_stupid, self).__init__()
        self.conv1 = nn.Conv2d(in_maps, 256, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        #self.conv3 = nn.Conv2d(128, 128, kernel_size=5, padding=2)

        self.pool = nn.AvgPool2d(kernel_size=(4, 1))
        self.pool2 = nn.AvgPool2d(kernel_size=(5, 1))
        self.conv4 = nn.Conv1d(256, N_COMP, kernel_size=5, padding=2)
        self.conv5 = nn.Conv1d(N_COMP, N_COMP, kernel_size=5, padding=2)
        self.activ = nn.ReLU()

        self.conv6 = nn.Conv2d(256, 256, kernel_size=(4, 1), padding=0)

        self.conv7 = nn.Conv2d(2*in_maps, 128, kernel_size=3, padding=1)
        self.conv8 = nn.Conv2d(128, 128, kernel_size=(4, 1), stride=(4, 1), padding=0)

    def forward(self, inp):
        # Assume inputs is simply x, shape n_batch x 1 x 128 x T
        #print (inp[1].shape, inp[2].shape, inp[3].shape)
        x2 = inp.transpose(2, 3)
        x2 = self.pool( self.activ( self.conv1(x2) ) )
        #print (x1.shape, x2.shape, x3.shape)
        x = self.pool( self.activ( self.conv2(x2) ) )
        #print (x.shape)
        x = self.pool2( self.activ( self.conv6(x) ) )
        x = torch.flatten(x, 2)
        x = self.activ( self.conv4(x) )
        output = self.activ( self.conv5(x) )
       
        
        return output 







if __name__ == '__main__':
    #inp = torch.zeros(16, 196)
    #inp2 = torch.zeros(2, 1, 28, 28)
    #f = SB_CNN(n_classes=10, in_maps=3)
    #H = HNet_AlexNet(N_COMP=30, T=227)
    H = HNet_FtEx_stupid(N_COMP=100, T=431)
    f = FtEx(n_classes=50)
    x = torch.rand([32, 1, 431, 128])
    output, inter = f(x)
    inter = x + 0.0
    H_x = H(inter)
    #W = NMF_D(FREQ=228)
    #h = explainer()
    #print(output.shape, inter.shape) 
    print (H_x.shape) 
    #print (W(H_x).shape)
    #print (h(H_x).shape) 
    #print (inter.shape)
    print ('All good')








