import h5py
from fastai.dataset import *


class SimulationHdf5Dataset(BaseDataset):
    """Custom Dataset for loading entries from HDF5 databases"""

    def __init__(self, h5_path, transform=None):
        self.h5f = h5py.File(h5_path, 'r',driver='core')
        super().__init__()
        self.transform = transform
        self.body_shape = self.h5f['bodies'].shape[1:]
        self.contact_shape = self.h5f['contacts'].shape[1:]
        self.body_channels=[self.h5f['body_channels'][i][0].decode('utf-8')
                            for i in range(self.h5f['body_channels'].shape[0])]
        self.contact_channels=[self.h5f['contact_channels'][i][0].decode('utf-8')
                               for i in range(self.h5f['contact_channels'].shape[0])]


#     def __getitem__(self, index):
        
#         features = self.h5f['bodies'][index]
#         label = self.h5f['contacts'][index]
#         if self.transform is not None:
#             import pdb; pdb.set_trace()
#             features = self.transform(features)
#         return features, label
    
    def get_n(self):
        return len(self.h5f['sim_fr'])
    
    def get_x(self, i):
        return self.h5f['bodies'][i]
    
    def get_y(self, i):
        return self.h5f['contacts'][i]
    
    def get_c(self):
        return 0
    
    def get_sz(self):
        return self.h5f['bodies'][0].shape[2]
    
    @property
    def is_multi(self): return False

    @property
    def is_reg(self): return True