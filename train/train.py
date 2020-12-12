import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from operator import itemgetter
import tqdm
import random
import matplotlib.pyplot as plt
# %matplotlib inline
import numpy as np

import struct
import math
from sys import byteorder

assert byteorder=='little', byteorder

def get_query_data(path_file):
    x = np.fromfile(path_file, dtype=np.uint64)[1:]
    datax = x[0::2] # key value
    datay = x[1::2] # index value
    return datax, datay

def get_index_data(path_file):
    print('reading index')
    res = np.fromfile(path_file, dtype=np.uint64)[1:]
    print('read done')
    return res

def convert(l1_model, l2_models, l2_error, out_path):
    '''
    l1model is one Cubic object, 
    l2_models is a list of Linear objects
    '''
    
    '''
    Encode VERSION_NUM = 0, num_layers = 2, # of models in l1, # of models in l2, 
           a,b,c,d, [a,b]*1000
    '''
    assert len(l1_model) == 1, 'layer 1 should only have one Cubic model'
    
    VERSION_NUM = 0
    num_layers = 2
    num_of_models_l1 = len(l1_model)
    num_of_models_l2 = len(l2_models)
    
    with open(out_path, 'wb') as fout:
        fout.write(struct.pack('<Q', VERSION_NUM))
        fout.write(struct.pack('<Q', num_layers))
        fout.write(struct.pack('<Q', 2))
        fout.write(struct.pack('<Q', 0))
        fout.write(struct.pack('<Q', num_of_models_l1))
        fout.write(struct.pack('<Q', num_of_models_l2))

        L0_params = np.array([l1_model[0].a, l1_model[0].b, l1_model[0].c, l1_model[0].d], dtype='<f8')
        # L1_params = []
        # L1_params = np.array(L1_params, dtype='<f8')

        # ba_l0 = bytearray()
        fout.write(L0_params.tobytes())
        print(L0_params)
        # ba_l1 = bytearray(struct.pack("f", L1_params))

        for x,e in zip(l2_models,l2_error):
            # L1_params.append(x.a)
            # L1_params.append(x.b)
            # L1_params.append(e)
            fout.write(struct.pack('<d', x.b.item()))
            fout.write(struct.pack('<d', x.a.item()))
            fout.write(struct.pack('<Q', int(math.ceil(e))))
            print(x.a.item(), x.b.item(), int(math.ceil(e)))
        # fout.write(L1_params.tobytes())

class Linear(nn.Module):
    
    def __init__(self):
        super(Linear, self).__init__()
        self.a = nn.Parameter(torch.tensor(0, dtype=torch.float64, requires_grad=True))
        self.b = nn.Parameter(torch.tensor(0, dtype=torch.float64, requires_grad=True))
        torch.nn.init.uniform_(self.a, -1, 1)
        torch.nn.init.uniform_(self.b, -1, 1)

    def forward(self, x):
        return x * self.a + self.b
    
    def transform_back(self, x1, x2, y1, y2):
        '''x1: shift for x
        x2: scale for x
        y1: shift for y
        y2: scale for y'''
        # original: f'(x')=a'x'+b'
        # transformed: f(x)=(f'(x')-y1)/y2
        # Thus f'(x')=f(x)*y2+y1=(ax+b)y2+y1=(a(x'-x1)/x2+b)y2+y1=(a/x2*x'-a/x2*x1+b)y2+y1
        #              = a/x2*y2 * x' + (-a/x2*x1+b)*y2+y1 = 
        # Now we're in transformed version. To return to original:
        # set a'=a/x2*y2, b'=(-a/x2*x1+b)*y2+y1
        a, b = self.a.item(), self.b.item()
        self.a.data.fill_(a/x2*y2)
        self.b.data.fill_((-a/x2*x1+b)*y2+y1)
    
class Cubic(nn.Module):
    
    def __init__(self):
        super(Cubic, self).__init__()
        self.a = nn.Parameter(torch.tensor(0, dtype=torch.float64, requires_grad=True))
        self.b = nn.Parameter(torch.tensor(0, dtype=torch.float64, requires_grad=True))
        self.c = nn.Parameter(torch.tensor(0, dtype=torch.float64, requires_grad=True))
        self.d = nn.Parameter(torch.tensor(0, dtype=torch.float64, requires_grad=True))
        torch.nn.init.uniform_(self.a, -1, 1)
        torch.nn.init.uniform_(self.b, -1, 1)
        torch.nn.init.uniform_(self.c, -1, 1)
        torch.nn.init.uniform_(self.d, -1, 1)
        
    def forward(self, x):
        return x**3 * self.a + x**2 * self.b + x * self.c + self.d
    
    def transform_back(self, x1, x2, y1, y2):
        '''x1: shift for x
        x2: scale for x
        y1: shift for y
        y2: scale for y'''
        
        a, b, c, d = self.a.item(), self.b.item(), self.c.item(), self.d.item()
        a_new = y2 * a / x2**3
        b_new = -3 * x1 * y2 * a / x2**3 + y2 * b / x2**2
        c_new = 3 * x1**2 * y2 * a / x2**3 - 2 * x1 * y2 * b / x2**2 + y2 * c / x2
        d_new = -x1**3 * y2 * a / x2**3 + x1**2 * y2 * b / x2**2 - x1 * y2 * c / x2 + d * y2 + y1
        self.a.data.fill_(a_new)
        self.b.data.fill_(b_new)
        self.c.data.fill_(c_new)
        self.d.data.fill_(d_new)


def L2_loss(output, target):
    loss = torch.mean((output - target)**2)
    return loss

def MaxLoss(output, target):
    return torch.max(torch.abs(output - target))

def transform(a):
    '''perform some form of standardization'''
    a = np.array(a)
    scale = max(a) - min(a)
    if scale == 0: scale = 1
    return (a - min(a)) / scale, min(a), scale

# evaluate model
def eval_model(model: nn.Module, x_val, y_val, criterion):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = len(x_val)
    loader = torch.utils.data.DataLoader(list(zip(x_val, y_val)), 
                                         batch_size=batch_size, 
                                         shuffle=True)
    with torch.no_grad():
        loss = 0
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            loss += criterion(model(data), target) * len(data)
        return (loss/len(x_val)).item()

def do_lr_decay(opt, epoch, lr_decay):
    lr = None
    for e, l in lr_decay:
        if epoch >= e: lr = l
    assert lr is not None, lr
#     print('update lr to', lr)
    for param_group in opt.param_groups:
        param_group['lr'] = lr

def train_model(model: nn.Module, x, y, 
                max_epoch=100, 
                criterion=L2_loss, 
                batch_size=None, 
                wd=0,
                lr_decay=((0,1e-1),),# ((0,1e-18), (40,1e-19), (70,1e-20)) #  lr=0.01 for epoch<4; lr=0.001 for epoch<7; ...
                log_freq=10,
               ):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x_ori, y_ori = x, y
    x, x_shift, x_scale = transform(x)
    y, y_shift, y_scale = transform(y)
    
    # Note: Alternative is optim.Adam optimizer, which claims to tune the lr automatically 
    # (though usually worse than hand-tuned SGD)
#     opt = optim.SGD(model.parameters(), lr=0.01, weight_decay=wd)
    opt = optim.Adam(model.parameters(), lr=0.1, weight_decay=wd)
    num_data = len(x)

    if batch_size is None: 
        batch_size = num_data # default to full batch SGD
    # create data loader to support mini-batch SGD
    train_loader = torch.utils.data.DataLoader(list(zip(x, y)), 
                                               batch_size=batch_size, 
                                               shuffle=True)
    for j in range(max_epoch):
        if log_freq > 0 and j % log_freq == 0: 
            print('Epoch', j, ': mean loss on training set is', eval_model(model, x, y, criterion))
        do_lr_decay(opt, j, lr_decay)
        for data, target in train_loader:
            # use GPU if available
            data, target = data.to(device), target.to(device)
            opt.zero_grad()
            loss = criterion(model(data), target)
            loss.backward()
            opt.step()
    err = eval_model(model, x, y, criterion)
    print('Final mean loss on training set is', err)

    # now transform model back: compute model' s.t. (model'(x_ori)-y_shift)/y_scale = model(x), 
    # where x is the transformed x_ori, i.e. x=(x_ori-x_shift)/x_scale
    model.transform_back(x_shift, x_scale, y_shift, y_scale)
    print(y_scale)

    err_ori = eval_model(model, x_ori, y_ori, criterion)
    print('Final mean original loss on training set is', err_ori)
    assert abs(err_ori / y_scale**2 - err) < 1e-5, (y_scale, err_ori / y_scale**2, err)

def seed_all(seed, deterministic_but_slow):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if deterministic_but_slow:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

def set_empty_const(empty_num, linear_list, data2_x, num_module2):
    # Empty set
    right_index = []
    left_index  = []
    const       = []
    if len(empty_num) == 1:
        right_index = empty_num[0] + 1
        left_index  = empty_num[0] - 1
        right_model = linear_list(right_index)
        right_val   = right_model.forward(data2_x[right_index][-1])
        left_model  = linear_list(left_index)
        left_val    = left_model.forward(data2_x[left_index][0])
        const.append(0.5 * (right_val + left_val))
    else:
        for i in range(len(empty_num)):

            signal_left  = -2
            signal_right = -2

            # Special Case: the first element
            if i == 0:
                if empty_num[i] == 0:
                    left_index.append(-1)
                else:
                    left_index.append(empty_num[i]-1)
                    signal_left = 0

                # Find the index of the first non-empty set at the left of the set whose index is empty_num(i)
                for k in range(i+1,len(empty_num)):
                    if empty_num[k] != empty_num[k-1] + 1:
                        right_index.append(empty_num[k-1]+ 1)
                        signal_right = 0
                        break
                if signal_right == -2:
                    right_index.append(-1)

            # Special Case: the last element
            elif i == len(empty_num)-1:
                if empty_num[i] == num_module2 - 1:
                    right_index.append(-1)
                else:
                    right_index.append(empty_num[i]+1)
                    signal_right = 0

                for l in range(i):
                    if empty_num[i-1-l] != empty_num[i-l] - 1:
                        left_index.append(empty_num[i-l] - 1)
                        signal_left = 0
                        break
                
                if signal_left == -2:
                    left_index.append(-1)
            
            # Usual Case
            else:


                # Find the index of the first non-empty set at the left of the set whose index is empty_num(i)
                for k in range(i+1,len(empty_num)):
                    if empty_num[k] != empty_num[k-1] + 1:
                        right_index.append(empty_num[k-1]+ 1)
                        signal_right = 0
                        break

                # Find the index of the first non-empty set at the right of the set whose index is empty_num(i)
                for l in range(i):
                    if empty_num[i-1-l] != empty_num[i-l] - 1:
                        left_index.append(empty_num[i-l] - 1)
                        signal_left = 0
                        break
                        
                if signal_right == -2:
                    right_index.append(-1)
                if signal_left == -2:
                    left_index.append(-1)

            if signal_right  == -2:
                left_model  = linear_list[left_index[i]]
                left_val    = left_model.forward(data2_x[left_index[i]][0])
                const.append(left_val.item())
            elif signal_left == -2:
                right_model = linear_list[right_index[i]]
                right_val   = right_model.forward(data2_x[right_index[i]][-1])
                const.append(right_val.item())
            else:
                right_model = linear_list[right_index[i]]
                right_val   = right_model.forward(data2_x[right_index[i]][-1])
                left_model  = linear_list[left_index[i]]
                left_val    = left_model.forward(data2_x[left_index[i]][0])
                const.append(0.5 * (right_val.item() + left_val.item()))
            
            linear_list[empty_num[i]].b = nn.Parameter(torch.tensor(const[i], dtype=torch.float64, requires_grad=True))


def train_L2(top_model, x, y, num_module2, log_freq=-1, max_epoch2=100, 
    criterion_train=L2_loss):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    linear_list = []
    errs = np.zeros(num_module2) # store max error

    # distibute data into 2nd layer
    with torch.no_grad():
        model_index = top_model(torch.tensor(x).to(device)).detach().cpu().numpy()
    print('model_index.shape=',model_index.shape)
    data2_x = [[] for _ in range(num_module2)]
    data2_y = [[] for _ in range(num_module2)]
    for i in range(len(model_index)):
        mi = max(0, min(num_module2 - 1, int(model_index[i])))
        data2_x[mi].append(x[i])
        data2_y[mi].append(y[i])
    del x,y # just for checking correctness

    # train 2nd layer
    linear_list  = []
    empty_num    = []
    print('num_data for each layer-2 model', list(map(len, data2_x)))
    for i in tqdm.tqdm(range(num_module2)):
        print(f'traing #{i}')
        linear_model = Linear().to(device)
        if len(data2_x[i]) == 0: 
            empty_num.append(i)
            linear_list.append(linear_model)
            continue # skip
        train_model(linear_model, data2_x[i], data2_y[i], max_epoch2, 
            log_freq=log_freq, criterion=criterion_train)
        linear_list.append(linear_model)
        max_err = eval_model(linear_model, data2_x[i], data2_y[i], MaxLoss)
        errs[i] = max_err
        print("max error={}".format(max_err))

    # compute max errors for empty models
    # first use left model's error
    for i in range(num_module2):
        if len(data2_x[i]) == 0 and i > 0: # model i is empty:
            errs[i] = errs[i - 1]
    # compute max(left model's error, right model's error)
    for i in reversed(range(num_module2)):
        if len(data2_x[i]) == 0 and i < num_module2 - 1: # model i is empty:
            errs[i] = max(errs[i], errs[i + 1])
    print(errs)
    set_empty_const(empty_num, linear_list, data2_x, num_module2)
    return linear_list, data2_x, data2_y, errs

def main():
    seed_all(7, True)
    #x           = np.arange(1000, dtype=np.float) # np.random.normal(0, 1, 1000)
    #y           = np.array([sum(x<inp) for inp in x]) #np.random.normal(0, 1, 1000)
    path_query  = "./data/wiki_ts_200M_uint64_queries_10M_in_train"
    path_index  = "./data/wiki_ts_200M_uint64"
    x, y        = get_query_data(path_query)
    max_epoch1  = 1000
    # max_epoch2  = 100
    num_module2 = 1000
    num_data1   = 1000 # try 1k first
    x, y        = x[:num_data1], y[:num_data1]
    nxt_lower_bound = np.sum(get_index_data(path_index) <= max(x)) # can be regarded as a rigorous alternative to max(datay)+1
    x, y        = x.astype(np.float64), y.astype(np.float64)
    cubic_list  = []
    datax       = x#x # 1st layer data
    datay       = y#y # 1st layer label
    datay       = (datay - min(datay)) * 1. / (nxt_lower_bound - min(datay)) * num_module2 #scale
    device      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log_freq    = -1
    out_path    = "./rmi_data/V1_3_PARAMETERS"
    print('init done')

    # train 1st layer
    cubic_list  = []
    print('num_data for each layer-1 model', len(datax))
    print(f'traing #{0}')
    cubic_model = Cubic().to(device)
    train_model(cubic_model, datax, datay, max_epoch1, log_freq=log_freq)
    cubic_list.append(cubic_model)

    linear_list, data2_x, data2_y, errs = train_L2(cubic_model, x.astype(np.float64), y.astype(np.float64), num_module2,
        log_freq=log_freq)
    wts = np.array(list(map(len, data2_x)))
    print("mean of max error of each layer 2 model=", sum(np.array(errs) * wts) / wts.sum())
    convert(cubic_list, linear_list, errs, out_path)
if __name__ == "__main__":
    main()
