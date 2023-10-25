
from scipy.linalg import orth
import torch
from torch.autograd import Variable
import copy
import numpy as np
from .utils import flatten_tensors, unflatten_tensors, ForkedPdb
from .evonorm import EvoNormSample2d as evonorm_s0
from collections import defaultdict

class CF_sender():
    def __init__(self, true_model, rank, device, classes):
        """
            Args
                model: the model on the sender device
                device: device on which the model is 
                include_norm: includes norm weights to gpm computation 
        """
        self.model           = copy.deepcopy(true_model)
        self.model           = self.model.to(device)
        self.features_buffer = None
        self.device          = device
        self.classes         = classes
        self.rank            = rank
        self.feature_size    = 0
        self.class_rep       = None
        self.model.eval()
        
    def _update_model(self, state_dict):
        """
            Args:
                state_dict: list of device parameters 
        """
        for w, p in zip(state_dict, self.model.parameters()):
                p.data.copy_(w.data)
        return


    def _accumulate_cross_features(self, x, features, y):
        """
            Args:
                x: inputs for which the gradients have to be accumulated
                targets: class labels for x
        """
        if features is None:
            features, _     = self.model(x)
        if self.feature_size == 0:
            self.feature_size = features.size(1)
        self.features_buffer = features.detach()   # size =  batch_size x hidden_nodes e.g., bx64 for resnet-20
        ####################################
        ##### compute average class features
        class_count   = torch.zeros(self.classes).to(self.device)
        self._clear_stack()
        for i in range(0, y.size(0)):
            self.class_rep[int(y[i])] += self.features_buffer[i].data
            #print(self.class_rep[int(y[i])].size(), self.features_buffer[i].size())
            class_count[int(y[i])]    += 1.0
        return self.features_buffer, self.class_rep, copy.deepcopy(class_count)

    def _clear_stack(self):
        if self.class_rep is None:
            self.class_rep   = torch.zeros(self.classes, self.feature_size).to(self.device)
        else:
            self.class_rep.data.mul_(0.0)

    def _flatten_(self, features):
        """
            Args
                features: Input to be flattened
            Returns
                flattened tensor
        """

        return flatten_tensors(features).to(self.device)

    def __call__(self, neighbor_weight, batch_x, self_features, y):
        """
            Args
                neighbor_weight: weights of the neighbor models
                batches: Input batches to compute variance
            Returns
                flattened gradients for each neighbor
        """

        cross_features = {}
        class_features = {}
        _, cr, count = self._accumulate_cross_features(None, self_features, y)
        class_features[self.rank] = copy.deepcopy(torch.cat([self._flatten_(cr), count]))
        for rank, w in neighbor_weight.items():
            self._update_model(w)
            f, cr, count = self._accumulate_cross_features(batch_x, None, y)
            cross_features[rank] = copy.deepcopy(f)
            class_features[rank] = copy.deepcopy(torch.cat([self._flatten_(cr), count]))
        return cross_features, class_features, self.feature_size

class CF_receiver():
    def __init__(self, device, rank, classes=10):
        self.rank           = rank
        self.device         = device
        self.classes        = classes
        self.class_features = None
        self.count          = None
        #self.pi            = 1.0/float(neighbors+1)      # !!! this has to updated. right now its hard coded for bidirectional ring topology with uniform weights
 
    def _unflatten_(self, flat_tensor, ref_buf):
        """
            Args
                flat_tensor: received flat tensor to be reshaped 
                ref_buf: reference buffer for computing unflattened shape
            Returns
                unflattened tensor based on reference tensor
        """
        count = flat_tensor[-int(self.classes):]
        unflat_tensor =  unflatten_tensors(flat_tensor[:-int(self.classes)], ref_buf)
        return unflat_tensor, count

    def __call__(self, self_cf, neighbor_cf, feature_size, y):
        """
            Args
                flat_tensor: received flat tensor to be reshaped 
                ref_buf: reference buffer for computing unflattened shape
            Returns
                computes the per class feature representation
        """
        ### Unflatten the neighbor grads
        
        ref_buf = torch.zeros(self.classes, feature_size).to(self.device)
        class_features, acc_count = self._unflatten_(self_cf, ref_buf)
        acc_cf                    = torch.stack(class_features)
        for rank, flat_tenor  in neighbor_cf.items():
            class_features, count = self._unflatten_(flat_tenor, ref_buf)
            acc_cf.data.add_(torch.stack(class_features))
            acc_count.data.add_(count.data)
        
        for i in range(self.classes):
            if acc_count[i]!=0:
                acc_cf[i].data.div_(acc_count[i].data)
        
        return acc_cf[y.long()]
                
