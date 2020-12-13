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
import argparse
import timer

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
        # ba_l1 = bytearray(struct.pack("f", L1_params))

        for x,e in zip(l2_models,l2_error):
            # L1_params.append(x.a)
            # L1_params.append(x.b)
            # L1_params.append(e)
            fout.write(struct.pack('<d', x.b.item()))
            fout.write(struct.pack('<d', x.a.item()))
            fout.write(struct.pack('<Q', int(math.ceil(e))))
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
    
def L1_loss(output, target):
    loss = torch.mean((output - target)**4)
    return loss

def MaxLoss(output, target):
    return torch.max(torch.abs(output - target))

def transform(a):
    '''perform some form of standardization'''
    a = np.array(a)
    scale = max(a) - min(a)
    if scale == 0: scale = 1
    b = (a - min(a)) / scale
    b = 2 * b - 1
    return b, min(a) + scale * 0.5, scale * 0.5

# evaluate model
def eval_model(model: nn.Module, x_val, y_val, criterion):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if isinstance(x_val, list): x_val = np.array(x_val)
    if isinstance(y_val, list): y_val = np.array(y_val)
    if isinstance(x_val, np.ndarray): x_val = torch.tensor(x_val).to(device)
    if isinstance(y_val, np.ndarray): y_val = torch.tensor(y_val).to(device)
    return criterion(model(x_val),y_val).item()

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
                lr_decay=((0,1),),# ((0,1e-18), (40,1e-19), (70,1e-20)) #  lr=0.01 for epoch<4; lr=0.001 for epoch<7; ...
                log_freq=10,
               ):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x_ori, y_ori = x, y
    x, x_shift, x_scale = transform(x)
    y, y_shift, y_scale = transform(y)
    x_gpu, y_gpu = torch.tensor(x).to(device), torch.tensor(y).to(device)
    
    # Note: Alternative is optim.Adam optimizer, which claims to tune the lr automatically 
    # (though usually worse than hand-tuned SGD)
    # opt = optim.SGD(model.parameters(), lr=0, weight_decay=wd)
    opt = optim.Adam(model.parameters(), lr=0, weight_decay=wd)
    num_data = len(x)
    assert batch_size is None # use full batch for now
    if batch_size is None: 
        batch_size = num_data # default to full batch SGD
    # create data loader to support mini-batch SGD
    # train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_gpu, y_gpu),
    #                                            batch_size=batch_size,
    #                                            shuffle=True if batch_size < len(x) else False,
    #                                            num_workers=8)
    train_loader = [(x_gpu, y_gpu)]
    train_loss = []
    for j in range(max_epoch):
        if log_freq > 0 and j % log_freq == 0: 
            train_loss.append((j, eval_model(model, x_gpu, y_gpu, criterion)))
            print('Epoch', j, ': mean loss on training set is', train_loss[-1][-1])
        do_lr_decay(opt, j, lr_decay)
        for data, target in train_loader:
            # use GPU if available
            data, target = data.to(device), target.to(device)
            opt.zero_grad()
            loss = criterion(model(data), target)
            loss.backward()
            opt.step()
    err = eval_model(model, x_gpu, y_gpu, criterion)
    train_loss.append((max_epoch, err))
    print('Final mean loss on training set is', err)

    # now transform model back: compute model' s.t. (model'(x_ori)-y_shift)/y_scale = model(x), 
    # where x is the transformed x_ori, i.e. x=(x_ori-x_shift)/x_scale
    model.transform_back(x_shift, x_scale, y_shift, y_scale)
    print(y_scale)

    err_ori = eval_model(model, x_ori, y_ori, criterion)
    print('Final mean original loss on training set is', err_ori)
    assert abs(err_ori / y_scale**2 - err) < 1e-5, (y_scale, err_ori / y_scale**2, err)
    return train_loss, y_scale

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

def set_empty_const(empty_num, linear_list, data2_y, num_module2):
    # Empty set
    right_index = []
    left_index  = []
    const       = []
    if len(empty_num) == 1:
        if empty_num[0] == 0:
            right_index = empty_num[0] + 1
            right_val   = sorted(data2_y[right_index])[0]
            const.append(right_val)
        elif empty_num[0] == num_module2 - 1:
            left_index  = empty_num[0] - 1
            left_val    = sorted(data2_y[left_index])[-1]
            const.append(left_val)
        else:
            right_index = empty_num[0] + 1
            left_index  = empty_num[0] - 1
            right_val   = sorted(data2_y[right_index])[0]
            left_val    = sorted(data2_y[left_index])[-1]
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
                    if k == len(empty_num)-1:
                        if empty_num[k] != num_module2 - 1:
                            right_index.append(empty_num[k] + 1)
                            signal_right = 0
                        break
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
                    if l == i-1:
                        if empty_num[i-1-l] != 0:
                            left_index.append(empty_num[i-1-l] - 1)
                            signal_left = 0
                        break
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
                    if k == len(empty_num)-1:
                        if empty_num[k] != num_module2 - 1:
                            right_index.append(empty_num[k] + 1)
                            signal_right = 0
                        break
                    if empty_num[k] != empty_num[k-1] + 1:
                        right_index.append(empty_num[k-1]+ 1)
                        signal_right = 0
                        break

                # Find the index of the first non-empty set at the right of the set whose index is empty_num(i)
                for l in range(i):
                    if l == i-1:
                        if empty_num[i-1-l] != 0:
                            left_index.append(empty_num[i-1-l] - 1)
                            signal_left = 0
                        break
                    if (empty_num[i-1-l] != empty_num[i-l] - 1):
                        left_index.append(empty_num[i-l] - 1)
                        signal_left = 0
                        break

                if signal_right == -2:
                    right_index.append(-1)
                if signal_left == -2:
                    left_index.append(-1)

            if signal_right  == -2:
                left_val    = sorted(data2_y[left_index[i]])[-1]
                const.append(left_val)
            elif signal_left == -2:
                right_val   = sorted(data2_y[right_index[i]])[0]
                const.append(right_val)
            else:
                right_val   = sorted(data2_y[right_index[i]])[0]
                left_val    = sorted(data2_y[left_index[i]])[-1]
                const.append(0.5 * (right_val.item() + left_val.item()))

            linear_list[empty_num[i]].b = nn.Parameter(torch.tensor(const[i], dtype=torch.float64, requires_grad=True))


def train_L2(top_model, x, y, num_module2, log_freq=-1, max_epoch2=100, 
    criterion_train=L1_loss):
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
    train_loss, y_scale = [[] for _ in range(num_module2)], [[] for _ in range(num_module2)]
    for i in tqdm.tqdm(range(num_module2)):
        print(f'traing #{i}')
        linear_model = Linear().to(device)
        if len(data2_x[i]) == 0: 
            empty_num.append(i)
            linear_list.append(linear_model)
            continue # skip
        train_loss[i], y_scale[i] = train_model(linear_model, data2_x[i], data2_y[i], max_epoch2, 
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
    return linear_list, data2_x, data2_y, errs, train_loss, y_scale

def do_stretch(x, y):
    assert np.all(x[:-1]<=x[1:]), 'data not sorted'
    # do the so-call "stretching" augmentation:
    # "given a key with position p before “stretching”, if its access frequency is
    # f, then we need to (1) shift its position to be p+ (f−1)/2; (2) and shift all keys after it with f−1"
        # the 2nd "shifting" part essentially equivallent to this (in my opinion, hope you can verify that)
    y1 = np.copy(y)
    for i in range(len(x)):
        if i == 0 or x[i-1] < x[i]:
            lb = i
        y1[i] = lb
    cnt_times = {}
    # 1st "shifting"
    for i in range(len(x)):
        cnt_times[x[i]]=cnt_times.get(x[i], 0) + 1 # calc frequency
    for i in range(len(x)):
        y1[i] += float(cnt_times[x[i]]-1) / 2 # shift to middle so that has same distance between previous key's y and next key's y
    return y1


def sort_data(x, y):
    idx = np.argsort(x)
    return x[idx], y[idx]

def work(x, y, index_array, out_path, max_epoch1, max_epoch2,
    num_module2, log_freq=-1, seed=7, deterministic_but_slow=True, stretch=False, args={}):
    ti = timer.Timer()
    # x, y        = get_query_data(path_query)
    # index_array = get_index_data(path_index)
    num_data1   = len(x)
    x, y        = sort_data(x, y)
    x, y        = x.astype(np.float64), y.astype(np.float64)
    cubic_list  = []
    datax       = x#x # 1st layer data
    datay       = y#y # 1st layer label
    if stretch:
        print('doing stretch')
        y1 = do_stretch(x, y)
        f = np.sum(y1 == max(y1))
        nxt_lower_bound = len(y1) # can be regarded as a rigorous alternative to max(datay)+1
        assert np.allclose(-(f-1.) / 2 + f + max(y1), nxt_lower_bound), nxt_lower_bound
        datay = y1
        datay = (datay - min(datay)) * 1. / (nxt_lower_bound - min(datay)) * num_module2 #scale
    else:
        nxt_lower_bound = np.sum(index_array <= max(x)) # can be regarded as a rigorous alternative to max(datay)+1
        datay       = (datay - min(datay)) * 1. / (nxt_lower_bound - min(datay)) * num_module2 #scale
    device      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    del index_array
    seed_all(seed, deterministic_but_slow)
    print('init done')

    # train 1st layer
    cubic_list  = []
    print('num_data for each layer-1 model', len(datax))
    ti('init')
    print(f'traing #{0}')
    cubic_model = Cubic().to(device)
    l1_train_loss, l1_y_scale = train_model(cubic_model, datax, datay, max_epoch1, log_freq=1)
    ti('train_l1')
    cubic_list.append(cubic_model)
    err1 = eval_model(cubic_model, datax, datay, L2_loss)

    ti('other')
    linear_list, data2_x, data2_y, errs, l2_train_loss, l2_y_scale = train_L2(cubic_model, x.astype(np.float64), y.astype(np.float64), num_module2,
        log_freq=log_freq, max_epoch2=max_epoch2)
    ti('train_l2')
    wts = np.array(list(map(len, data2_x)))
    mean_max_err = np.sum(np.array(errs) * wts) / wts.sum()
    print("mean of max error of each layer 2 model=", mean_max_err)
    convert(cubic_list, linear_list, errs, out_path)
    ti('other')
    np.savez(out_path+"_train_profile.npz", mean_max_err=mean_max_err, wts=wts, L2_err_layer1=err1, max_errs_layer2=errs, 
        linear_list=linear_list, cubic_list=cubic_list, loss={
            'l1_train_loss': l1_train_loss, 'l1_y_scale': l1_y_scale, 
            'l2_train_loss': l2_train_loss, 'l2_y_scale': l2_y_scale},
        ti=ti, args=vars(args))
    print(ti)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path-query', default="./data/wiki_ts_200M_uint64_queries_10M_in_train")
    parser.add_argument('--path-index', default="./data/wiki_ts_200M_uint64")
    parser.add_argument('--out-path', default="./rmi_data/V1_3_PARAMETERS")
    parser.add_argument('--max-epoch1', type=int, default=10000)
    parser.add_argument('--max-epoch2', type=int, default=1000)
    parser.add_argument('--num-module2', type=int, default=1000)
    parser.add_argument('--log-freq', type=int, default=100)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--stretch', action='store_true')
    args = parser.parse_args()
    path_query  = args.path_query
    path_index  = args.path_index
    out_path    = args.out_path
    max_epoch1 = args.max_epoch1
    max_epoch2 = args.max_epoch2
    num_module2 = args.num_module2
    log_freq = args.log_freq
    seed = args.seed
    stretch = args.stretch
    x, y = get_query_data(path_query)
    work(x, y, get_index_data(path_index), out_path, max_epoch1, max_epoch2, num_module2, 
        log_freq=log_freq, seed=seed, stretch=stretch, args=args)

if __name__ == "__main__":
    main()
