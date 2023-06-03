import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from data_preprocessing import data_load
import datetime
import time
import os
import csv
import copy
import argparse
import configparser
from torchsummaryX import summary
from sklearn.preprocessing import StandardScaler

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
torch.manual_seed(11)
np.random.seed(11)

device = torch.device('cuda:0')

args = argparse.ArgumentParser(description='args')
args.add_argument('--mode', default='train', type=str)
args.add_argument('--conf', type=str)
args_string = args.parse_args()
print(args)
try:
    config = configparser.ConfigParser()
    config.read(args_string.conf)
except:
    print("Config is not exist!")
    print(args_string.conf)
    exit()

#If you need to change the data, change the two lines below
args.add_argument('--data', default=config['data']['dataset'], type=str)
args.add_argument('--flowdata', default=config['data']['flowdata'], type=str)
args.add_argument('--weatherdata', default=config['data']['weatherdata'], type=str)
args.add_argument('--seg_num', default=config['data']['seg_num'], type=int)                 # d4: 307, metr-la: 207, bay: 325
args.add_argument('--use_weather', default=config['data']['use_weather'], type=int)
args.add_argument('--use_time', default=config['data']['use_time'], type=int)
args.add_argument('--saved_model', type=str)

#Model params
args.add_argument('--input_time', default=config['data']['input_time'], type=int)
args.add_argument('--input_feature', default=config['data']['input_feature'], type=int)
args.add_argument('--con_feature', default=config['model']['context_feature'], type=int)
args.add_argument('--emb_feature', default=config['model']['embed_feature'], type=int)
args.add_argument('--prediction_step', default=config['data']['prediction_step'], type=int)

#Hyper params
args.add_argument('--train_epoch', default=config['train']['train_epoch'], type=int)
args.add_argument('--batch_size', default=config['train']['batch_size'], type=int)
args.add_argument('--learning_rate', default=config['train']['learning_rate'], type=float)
args.add_argument('--dropout_ratio', default=config['train']['dropout_ratio'], type=float)
args.add_argument('--early_stop_patient', default=config['train']['early_stop_patient'], type=int)
args.add_argument('--use_gru', default=config['train']['use_gru'], type=eval)

args.add_argument("--steps", type=eval, default=[50, 100], help="steps")
args.add_argument("--lr_decay_ratio", type=float, default=0.1, help="lr_decay_ratio")
args.add_argument("--epsilon", type=float, default=1e-3, help="optimizer epsilon")

args = args.parse_args()

#args.saved_model = './out/data_Seoul_cityroad_511_ep_300_bs_8_lr_0.001_dr_0.5_er_20_bn_ln_220410_033125/best_model'
if args.saved_model:
    load_model = True
else:
    load_model = False

if args.mode=='test':
    t_epoch = 0
else:
    t_epoch=args.train_epoch


#Paths
log_file = 'log.txt'

datafile = './data/'+args.data+'.npy'
# wdatafile = './data/d4_time.npy'
wdatafile = './data/'+args.weatherdata+'.npy'
flowfile = './data/'+args.flowdata+'.npy'

#data_array, scaler = data_load(datafile, args.seg_num, args.input_time, args.prediction_step, args.input_feature)              ##########
scaler = StandardScaler()
data_array = np.load(datafile)
#data_array = np.swapaxes(data_array, 1, 0)

scaler.fit(data_array)
data_array_t = scaler.transform(data_array)

f_scaler = StandardScaler()
fdata_array = np.load(flowfile)[:-12] 
f_scaler.fit(fdata_array)
fdata_array = f_scaler.transform(fdata_array)
fdata_array = fdata_array.reshape((-1, args.seg_num, 1))

data_array = np.stack((data_array_t[:-12,:],data_array[12:,:]))
data_array = data_array.swapaxes(0, 2)
data_array = data_array.swapaxes(0, 1)
data_array = np.concatenate((data_array, fdata_array), axis=-1)
print(data_array.shape)
# data_array = np.reshape(data_array, [-1, args.seg_num, 2])
wdata_array_ = np.load(wdatafile, allow_pickle=True).astype('float32')
wdata_array = wdata_array_[:, :3]     # "In original version, [time index, temperatrue, rain fall]"

weather_scaler = StandardScaler()

wdata_array[:, 1] = weather_scaler.fit_transform(wdata_array[:, 1:2]).reshape(-1,)
wdata_array[:, 2] = weather_scaler.fit_transform(wdata_array[:, 2:3]).reshape(-1,)

# wdata_array = np.tile(wdata_array, (1,1,2))
wdata_array = np.stack((wdata_array_[:-12], wdata_array_[12:]))
wdata_array = wdata_array.swapaxes(0, 2)
wdata_array = wdata_array.swapaxes(0, 1)
wdata_array = np.concatenate((wdata_array, wdata_array_[:-12].reshape((-1, 3, 1))), axis=-1)
print(data_array.shape, wdata_array.shape)
data_array = np.concatenate((data_array, wdata_array), axis=1)
print(data_array.shape)

utcnow = datetime.datetime.utcnow()
now = utcnow + datetime.timedelta(hours=9)

if not os.path.isdir('./out/'):                                                           
    os.mkdir('./out/')
out_path = './out/data_{}_ep_{}_bs_{}_lr_{}_dr_{}_er_{}_bn_ln'.format(args.data,args.train_epoch, args.batch_size, args.learning_rate, args.dropout_ratio, args.early_stop_patient) + now.strftime('_%y%m%d_%H%M%S')
if not os.path.isdir(out_path):                                                           
    os.mkdir(out_path)

print('Parameters: \n data: {}\n epoch: {}\n batch: {}\n lr_rate: {}\n args.dropout_ratio: {}\n patient: {}\n'.format(args.data,args.train_epoch, args.batch_size, args.learning_rate, args.dropout_ratio, args.early_stop_patient))

#Our Traffic prediction Model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.node_cons = nn.Parameter(torch.FloatTensor(args.seg_num, args.con_feature))
        self.node_cons_f = nn.Parameter(torch.FloatTensor(args.seg_num, args.con_feature))
        #self.context_weight = nn.Parameter(torch.FloatTensor(args.con_feature, 1, args.emb_feature))
        #self.context_bias = nn.Parameter(torch.FloatTensor(args.con_feature, args.emb_feature))
        
        self.context_weight = nn.Parameter(torch.FloatTensor(args.con_feature, 2, args.emb_feature))
        self.context_bias = nn.Parameter(torch.FloatTensor(args.con_feature, args.emb_feature))

        self.context_weight_f = nn.Parameter(torch.FloatTensor(args.con_feature, 2, args.emb_feature))
        self.context_bias_f = nn.Parameter(torch.FloatTensor(args.con_feature, args.emb_feature))

        self.bn1 = nn.BatchNorm1d(args.input_time*args.seg_num)
        self.fbn1 = nn.BatchNorm1d(args.input_time*args.seg_num)
        self.bn2 = nn.BatchNorm1d(args.seg_num)
        self.bn3 = nn.BatchNorm1d(args.seg_num)
        self.bn4 = nn.BatchNorm1d(args.seg_num)

        # FC_input = (args.input_time*args.emb_feature*2)+12
        self.FC_input = (args.input_time*args.emb_feature*(2+args.use_weather))+12*args.use_time
        FC_input = self.FC_input
        self.FC = nn.Linear(FC_input, int(FC_input/3)).to(torch.float32)
        self.FC2 = nn.Linear(int(FC_input/3), args.prediction_step).to(torch.float32)
        self.FC3 = nn.Linear(args.prediction_step, args.prediction_step).to(torch.float32)
        #self.FC4 = nn.Linear(args.prediction_step*2, args.prediction_step)
        self.embFC = nn.Linear(1+args.use_time, args.emb_feature).to(torch.float32)
        self.fembFC = nn.Linear(1+args.use_time, args.emb_feature).to(torch.float32)

        self.wFC = nn.Linear(2, args.emb_feature).to(torch.float32)
        self.wbn = nn.BatchNorm1d(args.input_time).to(torch.float32)
        self.tFC = nn.Linear(12, args.emb_feature)
        self.tbn = nn.BatchNorm1d(args.seg_num)
        self.AttFC = nn.Linear(args.emb_feature,args.emb_feature)
        self.ReLU = nn.ReLU()
        self.telayer = nn.TransformerEncoderLayer(d_model=args.emb_feature, nhead=1, batch_first=True)
        self.te = nn.TransformerEncoder(self.telayer, 1)
        self.ftelayer = nn.TransformerEncoderLayer(d_model=args.emb_feature, nhead=1, batch_first=True)
        self.fte = nn.TransformerEncoder(self.ftelayer, 1)

        self.volume2speed = nn.MultiheadAttention(embed_dim=args.emb_feature, num_heads=1, dropout=0.5, batch_first=True)
        #self.re = Reformer(bucket_size=args.seg_num, dim=args.emb_feature, depth=1, heads=2)
        nn.init.kaiming_normal_(self.node_cons)
        nn.init.kaiming_normal_(self.context_weight)
        nn.init.kaiming_normal_(self.context_bias)
        nn.init.kaiming_normal_(self.node_cons_f)
        nn.init.kaiming_normal_(self.context_weight_f)
        nn.init.kaiming_normal_(self.context_bias_f)
        self.masking = torch.from_numpy(mask_(args.seg_num, args.input_time)).bool()

        self.gru = nn.GRU(input_size = args.seg_num , 
                        hidden_size = args.seg_num , 
                        dropout = 0.2, 
                        batch_first = True)
        self.be_gru = nn.Linear(args.seg_num, args.seg_num * 2).to(torch.float32)
        self.af_gru = nn.Linear(args.seg_num * 2, args.seg_num).to(torch.float32)
        #self.preFC = nn.Linear(FC_input * args.seg_num, args.seg_num).to(torch.float32)

    def forward(self, x, t, w, t_f, flow):
        x = x.to(device) #( batch, 12, road_num)

        origin_x = x

        #t = x[:,-1,:]
        #x = x[:,:-1,:]
        if args.use_time:
            #t = t.expand(x.shape[0], args.seg_num)
            #t = t.reshape(-1, args.seg_num, 1)
        
            t = t.repeat((1, args.seg_num, 1))
        
        #t = t.repeat_interleave(args.seg_num, dim=1)
        #t = torch.arange(0, args.seg_num).to(device).float()/args.seg_num
        #t = t.repeat((x.shape[0], args.input_time, 1))
        #t = torch.reshape(t, (-1, args.input_time*args.seg_num, 1))
        
        if args.use_time:
            x = torch.cat((x, t), axis=-1).float()
            flow = torch.cat((flow, t), axis=-1).float()
        
        # x = torch.reshape(x, (-1, args.input_time, args.seg_num, 2))
        # flow = torch.reshape(flow, (-1, args.input_time, args.seg_num, 2))
        # do = torch.nn.Dropout(p=args.dropout_ratio)
        # cwpl_weights = torch.einsum('ij,jkl->ikl', self.node_cons, self.context_weight)
        # cwpl_bias = self.node_cons.matmul(self.context_bias)
        # x = torch.einsum('btij,ijk->btik', x, cwpl_weights) + cwpl_bias
        
        # cwpl_weights_f = torch.einsum('ij,jkl->ikl', self.node_cons_f, self.context_weight_f)
        # cwpl_bias_f = self.node_cons_f.matmul(self.context_bias_f)
        # flow = torch.einsum('btij,ijk->btik', flow, cwpl_weights_f) + cwpl_bias_f
        x = self.embFC(x)
        flow = self.fembFC(flow)
        x = torch.reshape(x, (-1, args.input_time * args.seg_num, args.emb_feature))
        flow = torch.reshape(flow, (-1, args.input_time * args.seg_num, args.emb_feature))
        #x = torch.transpose(x, -1, 1)
        #print(x.shape)
        x = self.bn1(x)
        flow = self.fbn1(flow)
        #x = torch.transpose(x, -1, 1)
        x = F.gelu(x)
        flow = F.gelu(flow)
        #x = F.relu(x)
        #flow = F.relu(flow)
        #x = do(x)

        #x = torch.reshape(x, (-1, args.input_time * args.seg_num, args.emb_feature))
        output = self.te(x, mask=self.masking.to(device))
        #output = self.re(x, input_attn_mask=self.masking.to(device))
        #output = self.re(x)
        #output = torch.reshape(output, (-1, args.input_time, args.seg_num, args.emb_feature))
        #output = torch.transpose(output, 1, 2)
        #output = torch.reshape(output, (-1, args.seg_num, args.input_time * args.emb_feature))
        
        foutput = self.fte(flow, mask=self.masking.to(device))
        #foutput = torch.reshape(foutput, (-1, args.input_time, args.seg_num, args.emb_feature))
        #foutput = torch.transpose(foutput, 1, 2)
        #foutput = torch.reshape(foutput, (-1, args.seg_num, args.input_time * args.emb_feature))

        fusion_output = self.volume2speed(query=output, key=foutput, value=foutput, attn_mask=self.masking.to(device))[0]
        output = torch.cat((output, fusion_output), axis=2)
        output = torch.reshape(output, (-1, args.input_time, args.seg_num, args.emb_feature*2))
        output = torch.transpose(output, 1, 2)
        output = torch.reshape(output, (-1, args.seg_num, args.input_time * args.emb_feature*2))

        if args.use_weather:
            w = w.reshape((-1, args.input_time, 2)).to(torch.float32)

            w = self.wFC(w)
            w = self.wbn(w)
            w = F.gelu(w)
            #w = F.relu(w)
            w = w.reshape((-1, 1, args.input_time*args.emb_feature))
            w = w.repeat(1, args.seg_num, 1)
            output = torch.concat((output, w), axis=-1)
        
        if args.use_time:
            t_f = torch.transpose(t_f, 1, 2)
            t_f = t_f.repeat(1, args.seg_num, 1).to(torch.float32)
            #t_f = self.tFC(t_f)
            #t_f = self.tbn(t_f)
            output = torch.cat((output, t_f), axis=-1)
            # print(output.shape)

        if args.use_gru:
            output = output.view(-1, output.shape[1] * output.shape[2]) # (batch, road_num, emb_dim) > (batct, (road_num * emb_dim)) 
            output = self.preFC(output) # (batch, (road_num * emb_dim))  > (batch, road_num) 
            hidden = output.unsqueeze(dim=0) #(1, batch, road_num)
            # print(output.shape)
            # output = output.transpose(0, 1) # road_num * batch * 1
            # hidden = output.transpose(0, 2) # 1, batch, road_num
            
            output = []     
            
            out = origin_x.view(-1, 12, args.seg_num)
            out = out[:, -1:, :] # (batch, 1, road_num) // last timestep of input
            # out = self.be_gru(out) # (batch, 1, road_num * 2)
            
            for step in range(12):
                # print(out.shape, hidden.shape)
                out, hidden = self.gru(out, hidden)
                # real_out = self.af_gru(out)
                
                output.append(F.relu(out))
            
            output = torch.cat(output, dim=1)

        else:
            output = self.FC(output)
            #output = torch.transpose(output, -1, 1)
            output = self.bn2(output)
            #output = torch.transpose(output, -1, 1)
            output = F.gelu(output)
            #output = F.relu(output)
            #output = do(output)
            
            #output = torch.concat((output, t_f), axis=-1)
            
            output = self.FC2(output) # b * road_num * 12
            #output = self.bn3(output)
            output = F.relu(output) 

            output = torch.reshape(output, (-1, args.seg_num * args.prediction_step, 1)) 
            #output = torch.reshape(output, (-1, args.seg_num, args.prediction_step, 1)) 
            

        return output


def mask_(seg_num, input_time):
  masknp = np.empty((seg_num*input_time, seg_num*input_time))
  for i in range(input_time):
    tmp = np.empty((seg_num, input_time*seg_num))
    tmp[:, :(i+1)*seg_num] = False
    tmp[:, (i+1)*seg_num:] = True
    masknp[i*seg_num:(i+1)*seg_num, :] = tmp
  return masknp.astype('bool')


#Loss functions
def mape(pred, true, mask_value=0.):
    if mask_value != None:
        mask = torch.gt(true, mask_value)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)

    return torch.mean(torch.abs(torch.div((true - pred), true)))


def mae(pred, true, mask_value=0.):
    if mask_value != None:
        mask = torch.gt(true, mask_value)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
    return torch.mean(torch.abs(true-pred))


def rmse(pred, true, mask_value=0.):
    if mask_value != None:
        mask = torch.gt(true, mask_value)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
    return torch.sqrt(torch.mean((pred - true) ** 2))
##

def data_sliding(data_array):
    return np.lib.stride_tricks.sliding_window_view(data_array, 12, 0)

#Prepare training model
torch.cuda.empty_cache()
net = Net()
print(net)
print('Learnable parameters: '+str(sum(p.numel() for p in net.parameters() if p.requires_grad)))
net.to(device)
if load_model:
    net = torch.load(args.saved_model)
    net.to(device)
criterion = mae
# optimizer = optim.Adam(net.parameters(), lr=args.learning_rate)
optimizer = optim.Adam(net.parameters(), lr=args.learning_rate, eps=args.epsilon)
lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.steps, gamma=args.lr_decay_ratio)

##Data loading
data_array = data_sliding(data_array).swapaxes(-1, -3)
data_array = data_array.swapaxes(-1, -2)
print(data_array.shape)
#data_array = data_array[int(len(data_array)*0.75):]
dataloader = DataLoader(data_array[:int(len(data_array) * 0.7)], batch_size=args.batch_size, shuffle=True)
val_dataloader = DataLoader(data_array[int(len(data_array) * 0.7):int(len(data_array) * 0.8)], batch_size=args.batch_size, shuffle=False)
test_dataloader = DataLoader(data_array[int(len(data_array) * 0.8):], batch_size=args.batch_size, shuffle=False)


train_log = open(os.path.join(out_path, log_file), 'w', newline='') #logfile

wait = 0
val_mae_min = np.inf
best_model = copy.deepcopy(net.state_dict())
train_maes, val_maes, test_maes = [], [], []

#summary(net, torch.zeros((args.batch_size, args.input_time*args.seg_num, args.input_feature)))

#for epoch in range(0, t_epoch):
for epoch in range(0, args.train_epoch):

    if wait >= args.early_stop_patient:
        earlystop_log = 'Early stop at epoch: %04d' % (epoch)
        print(earlystop_log)
        train_log.write(earlystop_log + '\n')
        break
    
    mape_sum = 0
    loss_sum = 0
    mae_sum = 0
    rmse_sum = 0
    batch_num = 0
    cnt = 0
    net.train()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()

    for batch_idx, samples in enumerate(dataloader):
        if batch_idx % 400 == 0:
            print(batch_idx, len(dataloader))
        optimizer.zero_grad()
        input_x = samples[:, :, :args.seg_num, 0]
        input_x = np.reshape(input_x, [-1, args.seg_num * args.input_time, args.input_feature])
        input_t = samples[:, :, args.seg_num, 0]
        input_w = samples[:, :, args.seg_num+1:, 0]
        input_t = np.reshape(input_t, [-1, args.input_time, 1])
        input_w = np.reshape(input_w, [-1, args.input_time, 2])
        input_tf = samples[:, :, args.seg_num, 1]
        input_tf = np.reshape(input_tf, [-1, args.input_time, 1])
        input_f = samples[:, :, :args.seg_num, 2]
        input_f = np.reshape(input_f, [-1, args.seg_num * args.input_time, args.input_feature])
        labels = samples[:, :, :args.seg_num, 1] # b, 12, seg_num
        labels = np.swapaxes(labels, 1, 2)
        labels = np.reshape(labels, [-1, args.seg_num * args.prediction_step])
        input_x, labels, input_t, input_w, input_tf, input_f = input_x.to(device), labels.to(device), input_t.to(device), input_w.to(device), input_tf.to(device), input_f.to(device)

        outputs = net(input_x, input_t, input_w, input_tf, input_f)
        
        outputs = torch.reshape(outputs, (-1, args.seg_num * args.prediction_step))
        # outputs = torch.reshape(outputs, (-1, args.seg_num, args.prediction_step))
        # labels = torch.reshape(labels, (-1, args.seg_num, args.prediction_step))
        # outputs = outputs[:,:,args.prediction_step-1]
        # labels = labels[:,:,args.prediction_step-1]

        loss = criterion(outputs, labels)
        loss_sum += loss.item()
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 5)
        optimizer.step()

        #outputs = torch.reshape(outputs, (-1, args.seg_num)).cpu().detach().numpy()
        #outputs = torch.from_numpy(scaler.inverse_transform(outputs)).to(device)
        #labels = torch.reshape(labels, (-1, args.seg_num)).cpu().detach().numpy()
        #labels = torch.from_numpy(scaler.inverse_transform(labels)).to(device)
        #print(mae(outputs, labels))
        mae_sum += np.nan_to_num(mae(outputs, labels.to(device)).item())
        rmse_sum += np.nan_to_num(rmse(outputs, labels.to(device)).item())
        mape_step = np.nan_to_num(mape(outputs, labels.to(device)).item())
        mape_sum += mape_step
    
        batch_num = batch_idx
    batch_num += 1

    lr_scheduler.step()
    end.record()
    torch.cuda.synchronize()

    #Logging train, val(every) / test(every 5epoch) MAE loss
    ##Train_loss
    train_mae = mae_sum / batch_num
    train_maes.append(train_mae)
    train_log_str = ' train: %.5f\t\t%.5f\t\t%.5f' % (train_mae, rmse_sum / batch_num, mape_sum / batch_num)
    
    print('Epoch: ', epoch, ' / learning time: ', start.elapsed_time(end))
    print(train_log_str)
    train_log.write('Epoch: ' + str(epoch) + '\n')
    train_log.write(train_log_str + '\n')
    
    ##Val_loss
    with torch.no_grad():
        net.eval()
        loss_sum = 0
        mape_sum = 0
        rmse_sum = 0
        mae_sum = 0
        cnt = 0
        for batch_idx, samples in enumerate(val_dataloader):
            optimizer.zero_grad()
            input_x = samples[:, :, :args.seg_num, 0]
            input_x = np.reshape(input_x, [-1, args.seg_num * args.input_time, args.input_feature])
            input_t = samples[:, :, args.seg_num, 0]
            input_w = samples[:, :, args.seg_num+1:, 0]
            input_t = np.reshape(input_t, [-1, args.input_time, 1])
            input_w = np.reshape(input_w, [-1, args.input_time, 2])
            input_tf = samples[:, :, args.seg_num, 1]
            input_tf = np.reshape(input_tf, [-1, args.input_time, 1])
            input_f = samples[:, :, :args.seg_num, 2]
            input_f = np.reshape(input_f, [-1, args.seg_num * args.input_time, args.input_feature])
            labels = samples[:, :, :args.seg_num, 1]
            labels = np.swapaxes(labels, 1, 2)
            labels = np.reshape(labels, [-1, args.seg_num * args.prediction_step])
            input_x, labels, input_t, input_w, input_tf, input_f = input_x.to(device), labels.to(device), input_t.to(device), input_w.to(device), input_tf.to(device), input_f.to(device)
            outputs = net(input_x, input_t, input_w, input_tf, input_f)

            outputs = torch.reshape(outputs, (-1, args.seg_num * args.prediction_step))
            # outputs = torch.reshape(outputs, (-1, args.seg_num, args.prediction_step))
            # labels = torch.reshape(labels, (-1, args.seg_num, args.prediction_step))
            # outputs = outputs[:,:,args.prediction_step-1]
            # labels = labels[:,:,args.prediction_step-1]

            loss = criterion(outputs, labels)
            loss_sum += loss.item()
            #outputs = torch.reshape(outputs, (-1, args.seg_num)).cpu().detach().numpy()
            #outputs = torch.from_numpy(scaler.inverse_transform(outputs)).to(device)
            #labels = torch.reshape(labels, (-1, args.seg_num)).cpu().detach().numpy()
            #labels = torch.from_numpy(scaler.inverse_transform(labels)).to(device)
            mae_sum += np.nan_to_num(mae(outputs, labels.to(device)).item())
            rmse_sum += np.nan_to_num(rmse(outputs, labels.to(device)).item())
            mape_step = np.nan_to_num(mape(outputs, labels.to(device)).item())
            mape_sum += mape_step
            batch_num = batch_idx
        batch_num += 1
        
        val_mae = loss_sum / batch_num
        val_maes.append(val_mae)
        valid_log_str = ' Valid: %.5f\t\t%.5f\t\t%.5f' % (mae_sum / batch_num, rmse_sum / batch_num, mape_sum / batch_num)
        
        print(valid_log_str)
        train_log.write(valid_log_str + '\n')
        
        ##Model save
        #torch.save(net, os.path.join(out_path,'savedmodel_epoch_{}'.format(epoch)))
        
        #Early Stopping
        if val_mae <= val_mae_min:
            log = ' Validation loss decrease (exist min: %.5f, new min: %.5f)' % (val_mae_min, val_mae)
            print(log)
            train_log.write(log + '\n')
            best_model = copy.deepcopy(net.state_dict())
            wait=0
            val_mae_min = val_mae
        else:
            wait += 1
    
    ##Train loss (5 epoch)
    if epoch % 5 == 0:
        with torch.no_grad():
            net.eval()
            loss_sum = 0
            mape_sum = 0
            rmse_sum = 0
            mae_sum = 0
            cnt = 0
            #start = torch.cuda.Event(enable_timing=True)
            #end = torch.cuda.Event(enable_timing=True)
            #start.record()
            for batch_idx, samples in enumerate(test_dataloader):
                optimizer.zero_grad()
                input_x = samples[:, :, :args.seg_num, 0]
                input_x = np.reshape(input_x, [-1, args.seg_num * args.input_time, args.input_feature])
                input_t = samples[:, :, args.seg_num, 0]
                input_w = samples[:, :, args.seg_num+1:, 0] # n * 12 * 2
                input_t = np.reshape(input_t, [-1, args.input_time, 1])
                input_w = np.reshape(input_w, [-1, args.input_time, 2])
                input_tf = samples[:, :, args.seg_num, 1]
                input_tf = np.reshape(input_tf, [-1, args.input_time, 1])
                input_f = samples[:, :, :args.seg_num, 2]
                input_f = np.reshape(input_f, [-1, args.seg_num * args.input_time, args.input_feature])
                labels = samples[:, :, :args.seg_num, 1]
                labels = np.swapaxes(labels, 1, 2)
                labels = np.reshape(labels, [-1, args.seg_num * args.prediction_step])
                input_x, labels, input_t, input_w, input_tf, input_f = input_x.to(device), labels.to(device), input_t.to(device), input_w.to(device), input_tf.to(device), input_f.to(device)
                outputs = net(input_x, input_t, input_w, input_tf, input_f)

                outputs = torch.reshape(outputs, (-1, args.seg_num * args.prediction_step))
                # outputs = torch.reshape(outputs, (-1, args.seg_num, args.prediction_step))
                # labels = torch.reshape(labels, (-1, args.seg_num, args.prediction_step))
                # outputs = outputs[:,:,args.prediction_step-1]
                # labels = labels[:,:,args.prediction_step-1]

                loss = criterion(outputs, labels)
                loss_sum += loss.item()
                #outputs = torch.reshape(outputs, (-1, args.seg_num)).cpu().detach().numpy()
                #outputs = torch.from_numpy(scaler.inverse_transform(outputs)).to(device)
                #labels = torch.reshape(labels, (-1, args.seg_num)).cpu().detach().numpy()
                #labels = torch.from_numpy(scaler.inverse_transform(labels)).to(device)
                mae_sum += np.nan_to_num(mae(outputs, labels.to(device)).item())
                rmse_sum += np.nan_to_num(rmse(outputs, labels.to(device)).item())
                mape_step = np.nan_to_num(mape(outputs, labels.to(device)).item())
                mape_sum += mape_step
                batch_num = batch_idx
            batch_num += 1
            #end.record()
            #print('Test time:'+str(start.elapsed_time(end)))
            test_mae = mae_sum / batch_num
            test_maes.append(test_mae)
            test_log_str = ' test: %.5f\t\t%.5f\t\t%.5f' % (test_mae, rmse_sum / batch_num, mape_sum / batch_num)
            print(test_log_str)
            train_log.write(test_log_str + '\n')
            

#Logging last test mae loss
net.load_state_dict(best_model)
with torch.no_grad():
    net.eval()
    loss_sum = 0
    mape_sum = 0
    rmse_sum = 0
    mae_sum = 0
    cnt = 0
    x_np = np.array([])
    y_np = np.array([])
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for batch_idx, samples in enumerate(test_dataloader):
        optimizer.zero_grad()
        input_x = samples[:, :, :args.seg_num, 0]
        input_x = np.reshape(input_x, [-1, args.seg_num * args.input_time, args.input_feature])
        
        input_t = samples[:, :, args.seg_num, 0]
        input_w = samples[:, :, args.seg_num+1:, 0]
        input_t = np.reshape(input_t, [-1, args.input_time, 1])
        input_w = np.reshape(input_w, [-1, args.input_time, 2])
        
        input_tf = samples[:, :, args.seg_num, 1]
        input_tf = np.reshape(input_tf, [-1, args.input_time, 1])
        
        input_f = samples[:, :, :args.seg_num, 2]
        input_f = np.reshape(input_f, [-1, args.seg_num * args.input_time, args.input_feature])
        
        labels = samples[:, :, :args.seg_num, 1]
        labels = np.swapaxes(labels, 1, 2)
        labels = np.reshape(labels, [-1, args.seg_num * args.prediction_step])
        
        input_x, labels, input_t, input_w, input_tf, input_f = input_x.to(device), labels.to(device), input_t.to(device), input_w.to(device), input_tf.to(device), input_f.to(device)

        outputs = net(input_x, input_t, input_w, input_tf, input_f)
        outputs = torch.reshape(outputs, (-1, args.seg_num * args.prediction_step))        
        # outputs = torch.reshape(outputs, (-1, args.seg_num, args.prediction_step))
        # labels = torch.reshape(labels, (-1, args.seg_num, args.prediction_step))
        # outputs = outputs[:,:,args.prediction_step-1]
        # labels = labels[:,:,args.prediction_step-1]

        loss = criterion(outputs, labels)
        loss_sum += loss.item()
        #outputs = torch.reshape(outputs, (-1, args.seg_num)).cpu().detach().numpy()
        #outputs = torch.from_numpy(scaler.inverse_transform(outputs)).to(device)
        #labels = torch.reshape(labels, (-1, args.seg_num)).cpu().detach().numpy()
        #labels = torch.from_numpy(scaler.inverse_transform(labels)).to(device)
        mae_sum += np.nan_to_num(mae(outputs, labels.to(device)).item())
        rmse_sum += np.nan_to_num(rmse(outputs, labels.to(device)).item())
        mape_step = np.nan_to_num(mape(outputs, labels.to(device)).item())
        mape_sum += mape_step
        batch_num = batch_idx
    batch_num += 1
    test_log_str = ' Test: %.5f\t\t%.5f\t\t%.5f' % (mae_sum / batch_num, rmse_sum / batch_num, mape_sum / batch_num)
    print(test_log_str)
    end.record()
    train_log.write(test_log_str+'\n')
    #np.save('HY_pred', x_np)
    #np.save('HY_real', y_np)
    #print('Testinng time: ', start.elapsed_time(end))
    torch.save(net, os.path.join(out_path,'best_model'))
    print('Best model saved')

#Logging top 3 val/test mae loss
if t_epoch > 3:
    val_top3 = sorted(zip(val_maes, range(len(val_maes))))[:3]
    test_top3 = sorted(zip(test_maes, [i * 5 for i in range(len(test_maes))]))[:3]
    val_top3_log = \
        'Validation top 3\n 1st: %.5f / %depoch\n 2st: %.5f / %depoch\n 3st: %.5f / %depoch' % (val_top3[0][0], val_top3[0][1], val_top3[1][0], val_top3[1][1], val_top3[2][0], val_top3[2][1])
    test_top3_log = \
        'Test top 3\n 1st: %.5f / %depoch\n 2st: %.5f / %depoch\n 3st: %.5f / %depoch' % (test_top3[0][0], test_top3[0][1], test_top3[1][0], test_top3[1][1], test_top3[2][0], test_top3[2][1])

    print(val_top3_log)
    print(test_top3_log)
    train_log.write(val_top3_log + '\n')
    train_log.write(test_top3_log + '\n')
