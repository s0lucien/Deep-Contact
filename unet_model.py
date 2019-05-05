from fastai.conv_learner import *


class StdBlock(nn.Module):
    def __init__(self, nin, nout=None):
        super().__init__()
        self.nin=nin
        self.nout = nin if nout is None else nout
        self.conv = nn.Conv2d(self.nin, self.nout, kernel_size=3,padding=1)
        self.bn = nn.BatchNorm2d(self.nout)
    
    def forward(self, x): return self.bn(F.relu(self.conv(x)))

class StdDownsample(nn.Module):
    def __init__(self, nin):
        super().__init__()
        self.nin=nin
        self.pool = nn.AvgPool2d(2, padding=0)
        self.bn = nn.BatchNorm2d(nin)
        
    def forward(self, x): return self.bn(self.pool(x))

class StdUpsample(nn.Module):
    def __init__(self, nin, nout):
        super().__init__()
        self.nin=nin
        self.nout=nout
        self.conv = nn.ConvTranspose2d(self.nin, self.nout, 2, stride=2)
        self.bn = nn.BatchNorm2d(self.nout)
        
    def forward(self, x): return self.bn(F.relu(self.conv(x)))
    
class DilationBlock(nn.Module):
    def __init__(self, nin, nout=None, ndil=3):
        super().__init__()
        self.convs=nn.ModuleList()
        self.nin=nin
        self.nout = nin if nout is None else nout
        for i in range(ndil):
            d=2**i
            l=StdBlock(self.nin,self.nout)
            l.conv.dilation=(d,d)
            l.conv.padding=(d,d)
            self.convs.append(l)
    
    def forward(self, x):
#         import pdb; pdb.set_trace()
        act=self.convs[0](x)
        for li in range(1,len(self.convs)):
            act.add_(self.convs[li](x))
#         test_equality = sum([l(x) for l in self.convs])
#         assert np.allclose(test_equality.data.numpy(),act.data.numpy())
        return act


class UNet(nn.Module):
    def __init__(self, nin, nout, sz=128, ndil=3):
        super().__init__()
        self.nin = nin
        self.nout= nout
        self.img_sz=sz
        self.n_dilations = ndil
        self.conv1=nn.Sequential(StdBlock(self.nin,32),StdBlock(32,32)) # sz 128
        self.down1=nn.Sequential(StdDownsample(32),StdBlock(32,64),StdBlock(64,64)) #sz 64
        self.down2=nn.Sequential(StdDownsample(64),StdBlock(64,128),StdBlock(128,128)) #sz 32
        self.down_dil_up=nn.Sequential(StdDownsample(128),StdBlock(128,512),DilationBlock(512)
                                      ,StdUpsample(512,256)) #sz 32
        self.up1=nn.Sequential(StdBlock(128+256,128),StdBlock(128,128),StdUpsample(128,128)) #sz 32
        self.up2=nn.Sequential(StdBlock(64+128,64),StdBlock(64,64),StdUpsample(64,64)) #sz 64
        self.conv2=nn.Sequential(StdBlock(32+64,32),StdBlock(32,32),StdBlock(32,self.nout)) #sz 128
    
    def forward(self, x):
        # the layers of the pyramid
        p1 = self.conv1(x) # sz 128
        p2 = self.down1(p1) # sz 64
        p3 = self.down2(p2) # sz 32
        x = self.down_dil_up(p3) # sz 32
        x = self.up1(torch.cat([x,p3],dim=1)) # sz 64
        x = self.up2(torch.cat([x,p2],dim=1)) # sz 128
        x = self.conv2(torch.cat([x,p1],dim=1))
        return x
