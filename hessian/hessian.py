import torch
import math
from torch.autograd import Variable
import numpy as np

from hessian.utils import group_product, group_product_, group_add, normalization, get_params_grad, hessian_vector_product, orthnormal

class hessian():
    """
    The class used to compute :
        i) the top 1 (n) eigenvalue(s) of the neural network
        ii) the trace of the entire neural network
        iii) the estimated eigenvalue density
    """

    def __init__(self, model, criterion, data=None, dataloader=None, cuda=True):
        """
        model: the model that needs Hessain information
        criterion: the loss function
        data: a single batch of data, including inputs and its corresponding labels
        dataloader: the data loader including bunch of batches of data
        """

        # make sure we either pass a single batch or a dataloader
        assert (data != None and dataloader == None) or (data == None and
                                                         dataloader != None)

        self.model = model.eval()  # make model is in evaluation model
        self.criterion = criterion

        if data != None:
            self.data = data
            self.full_dataset = False
        else:
            self.data = dataloader
            self.full_dataset = True

        if cuda:
            self.device = 'cuda'
        else:
            self.device = 'cpu'

        # pre-processing for single batch case to simplify the computation.
        if not self.full_dataset:
            self.inputs, self.targets = self.data
            if self.device == 'cuda':
                self.inputs, self.targets = self.inputs.cuda(
                ), self.targets.cuda()

            # if we only compute the Hessian information for a single batch data, we can re-use the gradients.
            outputs = self.model(self.inputs)
            loss = self.criterion(outputs, self.targets)
            loss.backward(create_graph=True)

        # this step is used to extract the parameters from the model
        params, gradsH, names = get_params_grad(self.model)
        self.params = params
        self.gradsH = gradsH  # gradient used for Hessian computation
        self.names  = names

    def dataloader_hv_product(self, v):

        device = self.device
        num_data = 0  # count the number of datum points in the dataloader

        THv = [torch.zeros(p.size()).to(device) for p in self.params
              ]  # accumulate result
        for inputs, targets in self.data:
            self.model.zero_grad()
            tmp_num_data = inputs.size(0)
            outputs = self.model(inputs.to(device))
            loss = self.criterion(outputs, targets.to(device))
            loss.backward(create_graph=True)
            params, gradsH = get_params_grad(self.model)
            self.model.zero_grad()
            Hv = torch.autograd.grad(gradsH,
                                     params,
                                     grad_outputs=v,
                                     only_inputs=True,
                                     retain_graph=False)
            THv = [
                THv1 + Hv1 * float(tmp_num_data) + 0.
                for THv1, Hv1 in zip(THv, Hv)
            ]
            num_data += float(tmp_num_data)

        THv = [THv1 / float(num_data) for THv1 in THv]
        eigenvalue = group_product(THv, v).cpu().item()
        return eigenvalue, THv

    def eigenvalues(self, maxIter=100, tol=1e-3, top_n=1):
        """
        compute the top_n eigenvalues using power iteration method
        maxIter: maximum iterations used to compute each single eigenvalue
        tol: the relative tolerance between two consecutive eigenvalue computations from power iteration
        top_n: top top_n eigenvalues will be computed
        """

        assert top_n >= 1

        device = self.device

        eigenvalues = []
        eigenvectors = []

        computed_dim = 0

        while computed_dim < top_n:
            eigenvalue = None
            v = [torch.randn(p.size()).to(device) for p in self.params
                ]  # generate random vector
            v = normalization(v)  # normalize the vector

            for i in range(maxIter):
                v = orthnormal(v, eigenvectors)
                self.model.zero_grad()

                if self.full_dataset:
                    tmp_eigenvalue, Hv = self.dataloader_hv_product(v)
                else:
                    Hv = hessian_vector_product(self.gradsH, self.params, v)
                    tmp_eigenvalue = group_product(Hv, v).cpu().item()

                v = normalization(Hv)

                if eigenvalue == None:
                    eigenvalue = tmp_eigenvalue
                else:
                    if abs(eigenvalue - tmp_eigenvalue) / (abs(eigenvalue) +
                                                           1e-6) < tol:
                        break
                    else:
                        eigenvalue = tmp_eigenvalue
            eigenvalues.append(eigenvalue)
            eigenvectors.append(v)
            computed_dim += 1

        return eigenvalues, eigenvectors


    def _get_conv_params(self, model):
        """
        get conv. layer parameters and corresponding gradients
        for test using.
        """
        params = []
        grads = []
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                if not 'shortcut' in name or not 'downsample' in name :
                    params.append(module.weight)
                    grads.append(0. if module.weight is None else module.weight.grad)
        return params, grads

    def cal_trace(self, maxIter=100):

        for i in range(maxIter):
            _trace = []
            self.model.zero_grad()
            v = [
                torch.randint_like(p, high=2, device=self.device)
                for p in self.conv_params
            ]
            # generate Rademacher random variables
            for v_i in v:
                v_i[v_i == 0] = -1

            if self.full_dataset:
                _, Hv = self.dataloader_hv_product(v)
            else:
                for g, p, v_ in zip(self.conv_gradsH, self.conv_params, v):
                    Hv_temp = hessian_vector_product([g], [p], [v_])
                    _trace.append(group_product(Hv_temp,[v_]).cpu().item())
            print('conv_iter: ', i)
            trace_per_p = [t_p + t_ for t_p, t_ in zip(trace_per_p, _trace)]  
            
        return trace_per_p
    
    def trace(self, maxIter=150, tol=1e-3):
        """
        compute the trace of hessian using Hutchinson's method
        
        Args : 
            maxIter: maximum iterations used to compute trace
            tol: the relative tolerance

        Return :
            avg_dict_trace_layer_origin : average hessian trace for parameter in each layer
            avg_dict_trace_layer : check the accuracy for origin method
        """
        device = self.device

        # original method for calculte the network hessian trace
        trace_per_p_origin = [0.]*len(self.params)
        # check for original method's accuracy
        trace_per_p = [0.]*len(self.params)
        
        # avg hessian trace for paramter in each layer
        avg_dict_trace_layer_origin = {}
        avg_dict_trace_layer = {}

        for i in range(maxIter):
            # print('main iter: ', i)

            _trace = []
            self.model.zero_grad()
            v = [
                torch.randint_like(p, high=2, device=device)
                for p in self.params
            ]
            # generate Rademacher random variables
            for v_i in v:
                v_i[v_i == 0] = -1

            if self.full_dataset:
                _, Hv = self.dataloader_hv_product(v)
            else:
                # origin
                Hv = hessian_vector_product(self.gradsH, self.params, v)
                # check
                for g , p, v_ in zip(self.gradsH, self.params, v):
                    Hv_temp = hessian_vector_product([g], [p], [v_])
                    _trace.append(group_product(Hv_temp,[v_]).cpu().item())
            
            # origin
            t, t_per_layer_ = group_product_(Hv, v)
            trace_per_p_origin = [t_o + t.cpu().item() for t_o, t in zip(trace_per_p_origin, t_per_layer_)]

            # check
            trace_per_p = [t_p + t_ for t_p, t_ in zip(trace_per_p, _trace)]  

            '''
            # early stop
            if abs(sum(trace_per_p_origin)/i - trace) / (abs(trace) + 1e-6) < tol:
                return avg_dict_trace_layer, avg_dict_trace_layer_origin
            else:
                trace = sum(trace_per_p_origin)/i
            '''
        # calculate average trace per model parmeters
        for n, p, trace, trace_origin in zip(self.names, self.params, trace_per_p, trace_per_p_origin):
            avg_dict_trace_layer_origin[n] = trace_origin/(maxIter*p.numel())
            avg_dict_trace_layer[n] = trace/(maxIter*p.numel())

        return avg_dict_trace_layer, avg_dict_trace_layer_origin