# In[]
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  1 17:02:59 2022

@author: gwyoo
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import torch
import torch.nn as nn
import torch.optim as optim
#from sklearn.metrics import mean_absolute_error
#from sklearn.metrics import r2_score

# ====== Prepare Dataset ====== #
data = pd.read_csv("simpleboxkor_model_IB v2.csv", na_values=['#VALUE!', '#DIV/0!'])

data = data[["Emissions", "MW", "MP", "Kow", "VP", "WS", "PECair", "PECwater", "PECnaturalsoil", "PECagriculturalsoil", "PECothersoil"]]

drop_idx = data[data['PECair'].isnull()].index
data = data.drop(drop_idx)

data['Emissions'] = np.log10(data['Emissions'])
data['Kow'] = np.log10(data['Kow'])
data['VP'] = np.log10(data['VP'])
data['WS'] = np.log10(data['WS'])

data['PECair'] = np.log10(data['PECair'])
data['PECwater'] = np.log10(data['PECwater'])
data['PECnaturalsoil'] = np.log10(data['PECnaturalsoil'])
data['PECagriculturalsoil'] = np.log10(data['PECagriculturalsoil'])
data['PECothersoil'] = np.log10(data['PECothersoil'])

index = list(range(len(data)))
random.shuffle(index)
#index_sample = random.sample(index, 10000)
shuffled_data = data.iloc[index,:]
divide_num = round(len(shuffled_data) * 0.99)
trn = shuffled_data.iloc[:divide_num,:] # 70%
val = shuffled_data.iloc[divide_num:,:] # 30%

# ====== Split Dataset into Train, Validation ======#
train_X, train_y = trn.iloc[:,:6], trn.iloc[:,6:11]
val_X, val_y = val.iloc[:,:6], val.iloc[:,6:11]         

DEVICE = torch.device('cuda')

# In[]
class MLPModel(nn.Module): # 원래조건 : (5,20) (20,15), (15,10), (10,1)
    def __init__(self): 
        super(MLPModel, self).__init__()
        self.linear1 = nn.Linear(6,50)
        self.actv1 = nn.ReLU()
        self.linear2 = nn.Linear(50,71)
        self.actv2 = nn.ReLU()
        self.linear3 = nn.Linear(71,54)
        self.actv3 = nn.ReLU()
        self.linear4 = nn.Linear(54,84)
        self.actv4 = nn.ReLU()
        self.linear5 = nn.Linear(84,5)

    def forward(self, x):
    # 인스턴스(샘플) x가 인풋으로 들어왔을 때 모델이 예측하는 y값을 리턴합니다.
        x = self.linear1(x)
        x = self.actv1(x)
        x = self.linear2(x)
        x = self.actv2(x)
        x = self.linear3(x)
        x = self.actv3(x)
        x = self.linear4(x)
        x = self.actv4(x)
        x = self.linear5(x)
        return x
    
reg_loss = nn.MSELoss().to(DEVICE)

model = MLPModel().to(DEVICE) # Model을 생성해줍니다.
print('{} parameters'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))

lr = 0.008277 # Learning Rate를 하나 정해줍니다.
optimizer = optim.Adamax(model.parameters(), lr=lr) # Optimizer를 생성해줍니다.

list_epoch = [] 
list_train_loss = []
list_val_loss = []
list_mae = []
list_mae_epoch = []


# In[]
epoch = 10000 # 학습 횟수(epoch)을 지정해줍시다.

for i in range(epoch):
    
    # ====== Train ====== #
    model.train() # model을 train 모드로 세팅합니다. 반대로 향후 모델을 평가할 때는 eval() 모드로 변경할 겁니다 (나중 실습에서 쓸 겁니다)
    optimizer.zero_grad() # optimizer에 남아있을 수도 있는 잔여 그라디언트를 0으로 다 초기화해줍니다.
    
    input_x = torch.Tensor(train_X.values).to(DEVICE)
    true_y = torch.Tensor(train_y.values).to(DEVICE)
    pred_y = model(input_x)
    #print(input_x.shape, true_y.shape, pred_y.shape) # 각 인풋과 아웃풋의 차원을 체크해봅니다.
    
    loss = reg_loss(pred_y.squeeze(), true_y)
    #loss = reg_loss(torch.log10(pred_y.squeeze() + 1), torch.log10(true_y + 1)) # msle 로그MSE 최소화
    
    loss.backward() # backward()를 통해서 그라디언트를 구해줍니다.
    optimizer.step() # step()을 통해서 그라디언트를 바탕으로 파라미터를 업데이트 해줍니다.
    list_epoch.append(i)
    list_train_loss.append(loss.detach().cpu().numpy())
    
    
    # ====== Validation ====== #
    model.eval()
    optimizer.zero_grad()
    input_x = torch.Tensor(val_X.values).to(DEVICE)
    true_y = torch.Tensor(val_y.values).to(DEVICE)
    pred_y = model(input_x)
    loss = reg_loss(pred_y.squeeze(), true_y)
    
    #l2_lambda = 0.001
    #l2_norm = sum(p.pow(2.0).sum()
    #              for p in model.parameters())
 
    #loss = loss + l2_lambda * l2_norm
    
    list_val_loss.append(loss.detach().cpu().numpy())
    

    # ====== Evaluation ======= #
    if i % 200 == 0: # 200회의 학습마다 실제 데이터 분포와 모델이 예측한 분포를 그려봅니다.
        
        # ====== Calculate MAE ====== #
        #model.eval()
        #optimizer.zero_grad()
        #input_x = torch.Tensor(test_X.values)
        #true_y = torch.Tensor(test_y.values)
        #pred_y = model(input_x).detach().numpy() 
        #mae = mean_absolute_error(true_y, pred_y) # sklearn 쪽 함수들은 true_y 가 먼저, pred_y가 나중에 인자로 들어가는 것에 주의합시다
        #list_mae.append(mae)
        #list_mae_epoch.append(i)
        
        print(i, loss)
        
    if loss < 0.0045:
        print(i, loss, 'break')
        break
# In[]
torch.save(model.state_dict(), 'C:/Users/gwyoo/sbox_DNN_model/sbox_IB.pth') # 모델 저장

#model.load_state_dict(torch.load('C:\\Users\\juvox\\model\\sbox.pth')) # 모델 불러오기
# In[]
fig = plt.figure(dpi=300, figsize=(8,4))

# ====== Loss Fluctuation ====== # score 떨어지는 그림 그리기
ax1 = fig.add_subplot(1, 2, 1)
ax1.plot(list_epoch, list_train_loss, label='train_loss')
ax1.plot(list_epoch, list_val_loss, '--', label='val_loss')
ax1.set_xlabel('epoch')
ax1.set_ylabel('loss')
ax1.set_ylim(0, 10)
ax1.grid()
ax1.legend()
#ax1.set_title('epoch vs loss')

# ====== Metric Fluctuation ====== #
#ax2 = fig.add_subplot(1, 2, 2)
#ax2.plot(list_mae_epoch, list_mae, marker='x', label='mae metric')

#ax2.set_xlabel('epoch')
#ax2.set_ylabel('mae')
#ax2.grid()
#ax2.legend()
#ax2.set_title('epoch vs mae')
# In[]
z = list_train_loss
z = pd.DataFrame(z)
z.to_csv("list_train_loss.csv")
y = list_val_loss
y = pd.DataFrame(y)
y.to_csv("list_val_loss.csv")

plt.show()

# In[] model load
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import torch
import torch.nn as nn
import torch.optim as optim

class MLPModel(nn.Module): # 원래조건 : (5,20) (20,15), (15,10), (10,1)
    def __init__(self): 
        super(MLPModel, self).__init__()
        self.linear1 = nn.Linear(6,50)
        self.actv1 = nn.ReLU()
        self.linear2 = nn.Linear(50,71)
        self.actv2 = nn.ReLU()
        self.linear3 = nn.Linear(71,54)
        self.actv3 = nn.ReLU()
        self.linear4 = nn.Linear(54,84)
        self.actv4 = nn.ReLU()
        self.linear5 = nn.Linear(84,5)

    def forward(self, x):
    # 인스턴스(샘플) x가 인풋으로 들어왔을 때 모델이 예측하는 y값을 리턴합니다.
        x = self.linear1(x)
        x = self.actv1(x)
        x = self.linear2(x)
        x = self.actv2(x)
        x = self.linear3(x)
        x = self.actv3(x)
        x = self.linear4(x)
        x = self.actv4(x)
        x = self.linear5(x)
        return x

device = torch.device('cpu')   
model = MLPModel()
model.load_state_dict(torch.load('C:\\Users\\juvox\\model\\sbox.pth', map_location=device))
#model(train_input_x)

# In[] # training data prediction value output
data_X, data_y = data.iloc[:,:6], data.iloc[:,6:11]

data_input_x = torch.Tensor(data_X.values).to(DEVICE)
data_true_y = torch.Tensor(data_y.values).to(DEVICE)

data_reshape_NN = model(data_input_x)
# In[]

plt.rc('xtick', labelsize=15)
plt.rc('ytick', labelsize=15)
font = {'family' : 'times new roman',
        'weight' : 'bold',
        'size'   : 20}
plt.rc('font', **font)
plt.rc('axes', linewidth=2)
plt.rc('figure',facecolor = 'white')
title_font = {'fontname':'times new roman', 'size':'20', 'color':'black', 'weight':'normal',
  'verticalalignment':'bottom'} # Bottom vertical alignment for more space
axis_font = {'fontname':'times new roman', 'size':'20'}


plt.rc('figure', dpi=300, figsize=(15, 15))
#plt.figure(figsize=(12,12))
plt.rc('axes', linewidth=2)
plt.rcParams["font.family"] = "Times New Roman"
plotRange = 3
# In[]
plt.subplot(221)
plt.axis([-13, 3, -13, 3])
#plt.axis([-3, plotRange, -3, plotRange])
#plt.title("Deep learning model")
plt.xlabel("PNEC value (\u03bcg/L)")
plt.ylabel("DNN model Prediction (\u03bcg/L)")
_ = plt.plot(data_true_y.detach().cpu().numpy()[:,3], data_reshape_NN.detach().cpu().numpy()[:,3], 'ko', mfc = 'none', markersize = 3)
_ = plt.plot(np.arange(-13,plotRange + 1,1), np.arange(-13,plotRange + 1,1), linewidth = 1)
_ = plt.plot(np.log10( np.arange(10**(-13),10**(plotRange + 1),1) ), np.log10(2) + np.log10( np.arange(10**(-13),10**(plotRange + 1),1) ), linewidth = 1, color='r')
_ = plt.plot(np.log10( np.arange(10**(-13),10**(plotRange + 1),1) ), -np.log10(2) + np.log10( np.arange(10**(-13),10**(plotRange + 1),1) ), linewidth = 1, color='r')
#_ = plt.plot(np.arange(0,5,1), np.arange(0,5,1), linewidth = 1)

# In[]
plt.plot(data_true_y.detach().numpy()[:,1], data_reshape_NN.detach().numpy()[:,1], 'ko', mfc = 'none')
# In[] training data R2
SSE_NN = np.sum(np.square(data_true_y.detach().numpy()[:,0] - data_reshape_NN.detach().numpy()[:,0]))
SST = np.sum(np.square(data_true_y.detach().numpy()[:,0] - np.mean(data_true_y.detach().numpy()[:,0])))
rSquare_NN0 = (SST - SSE_NN) / SST

SSE_NN = np.sum(np.square(data_true_y.detach().numpy()[:,1] - data_reshape_NN.detach().numpy()[:,1]))
SST = np.sum(np.square(data_true_y.detach().numpy()[:,1] - np.mean(data_true_y.detach().numpy()[:,1])))
rSquare_NN1 = (SST - SSE_NN) / SST

SSE_NN = np.sum(np.square(data_true_y.detach().numpy()[:,2] - data_reshape_NN.detach().numpy()[:,2]))
SST = np.sum(np.square(data_true_y.detach().numpy()[:,2] - np.mean(data_true_y.detach().numpy()[:,2])))
rSquare_NN2 = (SST - SSE_NN) / SST

SSE_NN = np.sum(np.square(data_true_y.detach().numpy()[:,3] - data_reshape_NN.detach().numpy()[:,3]))
SST = np.sum(np.square(data_true_y.detach().numpy()[:,3] - np.mean(data_true_y.detach().numpy()[:,3])))
rSquare_NN3 = (SST - SSE_NN) / SST

SSE_NN = np.sum(np.square(data_true_y.detach().numpy()[:,4] - data_reshape_NN.detach().numpy()[:,4]))
SST = np.sum(np.square(data_true_y.detach().numpy()[:,4] - np.mean(data_true_y.detach().numpy()[:,4])))
rSquare_NN4 = (SST - SSE_NN) / SST


print("target 0 :", rSquare_NN0)
print("target 1 :", rSquare_NN1)
print("target 2 :", rSquare_NN2)
print("target 3 :", rSquare_NN3)
print("target 4 :", rSquare_NN4)
# In[]

a = data_reshape_NN.detach().numpy()
a = pd.DataFrame(a)
a.to_csv("data_reshape_NN(log).csv")
b = data_input_x.detach().numpy()
b = pd.DataFrame(b)
b.to_csv("data_input_x(log).csv")
c = data_true_y.detach().numpy()
c = pd.DataFrame(c)
c.to_csv("data_true_y(log).csv")

sw_MLR = torch.Tensor(sw_MLR.values)

plt.subplot(222)
plt.axis([0, plotRange, 0, plotRange])
#plt.title("Deep learning model")
plt.xlabel("PNEC value (\u03bcg/L)")
plt.ylabel("Multi-linear regression model Prediction (\u03bcg/L)")
_ = plt.plot(sw_test_true_y.detach().numpy(), sw_MLR.detach().numpy(), 'k^', mfc = 'none')
_ = plt.plot(np.arange(0,plotRange + 1,1), np.arange(0,plotRange + 1,1), linewidth = 1)

SSE_NN = np.sum(np.square(sw_test_true_y.detach().numpy() - sw_test_reshape_NN.detach().numpy()))
SST = np.sum(np.square(sw_test_true_y.detach().numpy() - np.mean(sw_test_true_y.detach().numpy())))
rSquare_NN = (SST - SSE_NN) / SST


SSE_MLR = np.sum(np.square(sw_test_true_y.detach().numpy() - sw_MLR.detach().numpy()))
rSquare_MLR = (SST - SSE_MLR) / SST
print("DNN model :", rSquare_NN, "MLR model : ", rSquare_MLR)



# In[] training set 그래프 # 
train_input_x = torch.Tensor(train_X.values)
train_true_y = torch.Tensor(train_y.values)
reshape_NN = torch.reshape(model(train_input_x), (-1,))

plt.rc('figure', dpi=300, figsize=(12, 12))
#plt.figure(figsize=(12,12))
plt.rc('axes', linewidth=2)
plt.rcParams["font.family"] = "Times New Roman"
plotRange = 300

plt.subplot(221)
plt.axis([0, plotRange, 0, plotRange])
#plt.title("Deep learning model")
plt.xlabel("PNEC value (\u03bcg/L)")
plt.ylabel("DNN model Prediction (\u03bcg/L)")
_ = plt.plot(train_true_y.detach().numpy(), reshape_NN.detach().numpy(), 'ko', mfc = 'none')
_ = plt.plot(np.arange(0,plotRange + 1,1), np.arange(0,plotRange + 1,1), linewidth = 1)

a = reshape_NN.detach().numpy()
a = pd.DataFrame(a)
a.to_csv("train_reshape_NN.csv")
b = train_input_x.detach().numpy()
b = pd.DataFrame(b)
b.to_csv("train_input_x.csv")
c = train_true_y.detach().numpy()
c = pd.DataFrame(c)
c.to_csv("train_true_y.csv")



MLR = trn.iloc[:,6]
MLR = torch.Tensor(MLR.values)

plt.subplot(222)
plt.axis([0, plotRange, 0, plotRange])
#plt.title("Deep learning model")
plt.xlabel("PNEC value (\u03bcg/L)")
plt.ylabel("Multi-linear regression model Prediction (\u03bcg/L)")
_ = plt.plot(train_true_y.detach().numpy(), MLR.detach().numpy(), 'k^', mfc = 'none')
_ = plt.plot(np.arange(0,plotRange + 1,1), np.arange(0,plotRange + 1,1), linewidth = 1)


SSE_NN = np.sum(np.square(train_true_y.detach().numpy() - reshape_NN.detach().numpy()))
SST = np.sum(np.square(train_true_y.detach().numpy() - np.mean(train_true_y.detach().numpy())))
rSquare_NN = (SST - SSE_NN) / SST


SSE_MLR = np.sum(np.square(train_true_y.detach().numpy() - MLR.detach().numpy()))
rSquare_MLR = (SST - SSE_MLR) / SST
print("DNN model :", rSquare_NN, "MLR model : ", rSquare_MLR)
# In[] # validation 그래프
val_input_x = torch.Tensor(val_X.values)
val_true_y = torch.Tensor(val_y.values)
val_reshape_NN = torch.reshape(model(val_input_x), (-1,))

plt.rc('figure', dpi=300, figsize=(12, 12))
#plt.figure(figsize=(12,12))
plt.rc('axes', linewidth=2)
plt.rcParams["font.family"] = "Times New Roman"
plotRange = 370

plt.subplot(221)
plt.axis([0, plotRange, 0, plotRange])
#plt.title("Deep learning model")
plt.xlabel("PNEC value (\u03bcg/L)")
plt.ylabel("DNN model Prediction (\u03bcg/L)")
_ = plt.plot(val_true_y.detach().numpy(), val_reshape_NN.detach().numpy(), 'ko', mfc = 'none')
_ = plt.plot(np.arange(0,plotRange + 1,1), np.arange(0,plotRange + 1,1), linewidth = 1)

val_MLR = val.iloc[:,6]
val_MLR = torch.Tensor(val_MLR.values)

plt.subplot(222)
plt.axis([0, plotRange, 0, plotRange])
#plt.title("Deep learning model")
plt.xlabel("PNEC value (\u03bcg/L)")
plt.ylabel("Multi-linear regression model Prediction (\u03bcg/L)")
_ = plt.plot(val_true_y.detach().numpy(), val_MLR.detach().numpy(), 'k^', mfc = 'none')
_ = plt.plot(np.arange(0,plotRange + 1,1), np.arange(0,plotRange + 1,1), linewidth = 1)

SSE_NN = np.sum(np.square(val_true_y.detach().numpy() - val_reshape_NN.detach().numpy()))
SST = np.sum(np.square(val_true_y.detach().numpy() - np.mean(val_true_y.detach().numpy())))
rSquare_NN = (SST - SSE_NN) / SST

SSE_MLR = np.sum(np.square(val_true_y.detach().numpy() - val_MLR.detach().numpy()))
rSquare_MLR = (SST - SSE_MLR) / SST
print(rSquare_NN, rSquare_MLR)

# In[] # kr test set 764개 그래프


# ====== Prepare Dataset ====== #

sbox_data = pd.read_csv("simplebox_ex - RB.csv")
sbox_data = sbox_data[["Emissions", "MW", "MP", "Kow", "VP", "WS","PECair", "PECwater", "PECnaturalsoil", "PECagriculturalsoil", "PECothersoil"]]


sbox_data['Emissions'] = np.log10(sbox_data['Emissions'])
sbox_data['Kow'] = np.log10(sbox_data['Kow'])
sbox_data['VP'] = np.log10(sbox_data['VP'])
sbox_data['WS'] = np.log10(sbox_data['WS'])


sbox_X, sbox_y = sbox_data.iloc[:,:6], sbox_data.iloc[:,6:11]

sbox_input_x = torch.Tensor(sbox_X.values)
sbox_true_y = torch.Tensor(sbox_y.values)

sbox_reshape_NN = model(sbox_input_x)
a = sbox_reshape_NN.detach().numpy()
a = pd.DataFrame(a)
result = np.power(10, a)

result.to_csv("sbox_reshape_NN(log).csv")



plt.rc('figure', dpi=300, figsize=(12, 12))
#plt.figure(figsize=(12,12))
plt.rc('axes', linewidth=2)
plt.rcParams["font.family"] = "Times New Roman"
plotRange = 150

plt.subplot(221)
plt.axis([0, plotRange, 0, plotRange])
#plt.title("Deep learning model")
plt.xlabel("PNEC value (\u03bcg/L)")
plt.ylabel("DNN model Prediction (\u03bcg/L)")
_ = plt.plot(sbox_true_y.detach().numpy(), sbox_reshape_NN.detach().numpy(), 'ko', mfc = 'none', markerfacecolor = 'r', markeredgecolor = 'k', label = 'DNN result')
_ = plt.plot(np.arange(0,plotRange + 1,1), np.arange(0,plotRange + 1,1), linewidth = 2, color = 'k')
plt.legend()

a = sbox_reshape_NN.detach().numpy()
a = pd.DataFrame(a)
a.to_csv("sbox_reshape_NN(log).csv")
b = kr_test_input_x.detach().numpy()
b = pd.DataFrame(b)
b.to_csv("sbox_input_x(log).csv")
c = kr_test_true_y.detach().numpy()
c = pd.DataFrame(c)
c.to_csv("sbox_true_y(log).csv")







kr_MLR = torch.Tensor(kr_MLR.values)

plt.subplot(222)
plt.axis([0, plotRange, 0, plotRange])
#plt.title("Deep learning model")
plt.xlabel("PNEC value (\u03bcg/L)")
plt.ylabel("Multi-linear regression model Prediction (\u03bcg/L)")
_ = plt.plot(kr_test_true_y.detach().numpy(), kr_MLR.detach().numpy(), 'k^', mfc = 'none',markerfacecolor = 'b', markeredgecolor = 'k', label = 'MLR result')
_ = plt.plot(np.arange(0,plotRange + 1,1), np.arange(0,plotRange + 1,1), linewidth = 2, color = 'k')
plt.legend()
SSE_NN = np.sum(np.square(kr_test_true_y.detach().numpy() - kr_test_reshape_NN.detach().numpy()))
SST = np.sum(np.square(kr_test_true_y.detach().numpy() - np.mean(kr_test_true_y.detach().numpy())))
rSquare_NN = (SST - SSE_NN) / SST


SSE_MLR = np.sum(np.square(kr_test_true_y.detach().numpy() - kr_MLR.detach().numpy()))
rSquare_MLR = (SST - SSE_MLR) / SST
print("DNN model :", rSquare_NN, "MLR model : ", rSquare_MLR)

# In[] # US + SW prediction and plot
ussw_test_df = pd.read_csv("US_SW(536).csv", delimiter=",")
us_test_df = ussw_test_df[(ussw_test_df['source']=="US EPA") | (ussw_test_df['source']== "Oregon") | (ussw_test_df['source']== "add_Oregon")]
us_test_data = us_test_df[["pH", "Cond", "DOC", "EU-Chronic", "MLR"]]
us_test_data['Cond'] = np.log10(us_test_data['Cond'])
us_test_data['DOC'] = np.log10(us_test_data['DOC'])

us_test_X, us_test_y = us_test_data.iloc[:,:3], us_test_data.iloc[:,3]
us_MLR = us_test_data.iloc[:,4]

us_test_input_x = torch.Tensor(us_test_X.values)
us_test_true_y = torch.Tensor(us_test_y.values)

us_test_reshape_NN = torch.reshape(model(us_test_input_x), (-1,))

plt.rc('figure', dpi=300, figsize=(12, 12))
#plt.figure(figsize=(12,12))
plt.rc('axes', linewidth=2)
plt.rcParams["font.family"] = "Times New Roman"
plotRange = 40

plt.subplot(221)
plt.axis([0, plotRange, 0, plotRange])
#plt.title("Deep learning model")
plt.xlabel("PNEC value (\u03bcg/L)")
plt.ylabel("DNN model Prediction (\u03bcg/L)")
_ = plt.plot(us_test_true_y.detach().numpy(), us_test_reshape_NN.detach().numpy(), 'ko', mfc = 'none')
_ = plt.plot(np.arange(0,plotRange + 1,1), np.arange(0,plotRange + 1,1), linewidth = 1)

a = us_test_reshape_NN.detach().numpy()
a = pd.DataFrame(a)
a.to_csv("us_test_reshape_NN(log).csv")
b = us_test_input_x.detach().numpy()
b = pd.DataFrame(b)
b.to_csv("us_test_input_x(log).csv")
c = us_test_true_y.detach().numpy()
c = pd.DataFrame(c)
c.to_csv("us_test_true_y(log).csv")
 
us_MLR = torch.Tensor(us_MLR.values)

plt.subplot(222)
plt.axis([0, plotRange, 0, plotRange])
#plt.title("Deep learning model")
plt.xlabel("PNEC value (\u03bcg/L)")
plt.ylabel("Multi-linear regression model Prediction (\u03bcg/L)")
_ = plt.plot(us_test_true_y.detach().numpy(), us_MLR.detach().numpy(), 'k^', mfc = 'none')
_ = plt.plot(np.arange(0,plotRange + 1,1), np.arange(0,plotRange + 1,1), linewidth = 1)

SSE_NN = np.sum(np.square(us_test_true_y.detach().numpy() - us_test_reshape_NN.detach().numpy()))
SST = np.sum(np.square(us_test_true_y.detach().numpy() - np.mean(us_test_true_y.detach().numpy())))
rSquare_NN = (SST - SSE_NN) / SST


SSE_MLR = np.sum(np.square(us_test_true_y.detach().numpy() - us_MLR.detach().numpy()))
rSquare_MLR = (SST - SSE_MLR) / SST
print("DNN model :", rSquare_NN, "MLR model : ", rSquare_MLR)

# In[] sw prediction
sw_test_df = ussw_test_df[ussw_test_df['source']=="SW"]
sw_test_data = sw_test_df[["pH", "Cond", "DOC", "EU-Chronic", "MLR"]]
sw_test_data['Cond'] = np.log10(sw_test_data['Cond'])
sw_test_data['DOC'] = np.log10(sw_test_data['DOC'])

sw_test_X, sw_test_y = sw_test_data.iloc[:,:3], sw_test_data.iloc[:,3]
sw_MLR = sw_test_data.iloc[:,4]

sw_test_input_x = torch.Tensor(sw_test_X.values)
sw_test_true_y = torch.Tensor(sw_test_y.values)

sw_test_reshape_NN = torch.reshape(model(sw_test_input_x), (-1,))

plt.rc('figure', dpi=300, figsize=(12, 12))
#plt.figure(figsize=(12,12))
plt.rc('axes', linewidth=2)
plt.rcParams["font.family"] = "Times New Roman"
plotRange = 120

plt.subplot(221)
plt.axis([0, plotRange, 0, plotRange])
#plt.title("Deep learning model")
plt.xlabel("PNEC value (\u03bcg/L)")
plt.ylabel("DNN model Prediction (\u03bcg/L)")
_ = plt.plot(sw_test_true_y.detach().numpy(), sw_test_reshape_NN.detach().numpy(), 'ko', mfc = 'none')
_ = plt.plot(np.arange(0,plotRange + 1,1), np.arange(0,plotRange + 1,1), linewidth = 1)

a = sw_test_reshape_NN.detach().numpy()
a = pd.DataFrame(a)
a.to_csv("sw_test_reshape_NN(log).csv")
b = sw_test_input_x.detach().numpy()
b = pd.DataFrame(b)
b.to_csv("sw_test_input_x(log).csv")
c = sw_test_true_y.detach().numpy()
c = pd.DataFrame(c)
c.to_csv("sw_test_true_y(log).csv")

sw_MLR = torch.Tensor(sw_MLR.values)

plt.subplot(222)
plt.axis([0, plotRange, 0, plotRange])
#plt.title("Deep learning model")
plt.xlabel("PNEC value (\u03bcg/L)")
plt.ylabel("Multi-linear regression model Prediction (\u03bcg/L)")
_ = plt.plot(sw_test_true_y.detach().numpy(), sw_MLR.detach().numpy(), 'k^', mfc = 'none')
_ = plt.plot(np.arange(0,plotRange + 1,1), np.arange(0,plotRange + 1,1), linewidth = 1)

SSE_NN = np.sum(np.square(sw_test_true_y.detach().numpy() - sw_test_reshape_NN.detach().numpy()))
SST = np.sum(np.square(sw_test_true_y.detach().numpy() - np.mean(sw_test_true_y.detach().numpy())))
rSquare_NN = (SST - SSE_NN) / SST


SSE_MLR = np.sum(np.square(sw_test_true_y.detach().numpy() - sw_MLR.detach().numpy()))
rSquare_MLR = (SST - SSE_MLR) / SST
print("DNN model :", rSquare_NN, "MLR model : ", rSquare_MLR)



# In[] # Bio-met_DB 3개 국가
BM_test_data = pd.read_csv("Biomet_DB.csv")
BM_test_data = BM_test_data[["pH", "Ca", "DOC", "Cu_HC5", "Biomet"]] 

BM_test_X, BM_test_y = BM_test_data.iloc[:,:3], BM_test_data.iloc[:,3]
BM_Biomet = BM_test_data.iloc[:,4]

BM_test_input_x = torch.Tensor(BM_test_X.values)
BM_test_true_y = torch.Tensor(BM_test_y.values)

BM_test_reshape_NN = torch.reshape(model(BM_test_input_x), (-1,))

plt.rc('figure', dpi=300, figsize=(12, 12))
#plt.figure(figsize=(12,12))
plt.rc('axes', linewidth=2)
plt.rcParams["font.family"] = "Times New Roman"
plotRange = 150

plt.subplot(221)
plt.axis([0, plotRange, 0, plotRange])
#plt.title("Deep learning model")
plt.xlabel("PNEC value (\u03bcg/L)")
plt.ylabel("DNN model Prediction (\u03bcg/L)")
_ = plt.plot(BM_test_true_y.detach().numpy(), BM_test_reshape_NN.detach().numpy(), 'ko', mfc = 'none', markerfacecolor = 'r', markeredgecolor = 'k', label = 'DNN result')
_ = plt.plot(np.arange(0,plotRange + 1,1), np.arange(0,plotRange + 1,1), linewidth = 2, color = 'k')
plt.legend()
a = BM_test_reshape_NN.detach().numpy()
a = pd.DataFrame(a)
a.to_csv("Biomet_DB_reshape_NN.csv")
b = BM_test_input_x.detach().numpy()
b = pd.DataFrame(b)
b.to_csv("Biomet_DB_input_x.csv")
c = BM_test_true_y.detach().numpy()
c = pd.DataFrame(c)
c.to_csv("Biomet_DB_true_y.csv")

BM_Biomet = torch.Tensor(BM_Biomet.values)

plt.subplot(222)
plt.axis([0, plotRange, 0, plotRange])
#plt.title("Deep learning model")
plt.xlabel("PNEC value (\u03bcg/L)")
plt.ylabel("Bio-Met model Prediction (\u03bcg/L)")
_ = plt.plot(BM_test_true_y.detach().numpy(), BM_Biomet.detach().numpy(), 'k^', mfc = 'none',markerfacecolor = 'b', markeredgecolor = 'k', label = 'Bio-met result')
_ = plt.plot(np.arange(0,plotRange + 1,1), np.arange(0,plotRange + 1,1), linewidth = 2, color = 'k')
plt.legend()
SSE_NN = np.sum(np.square(BM_test_true_y.detach().numpy() - BM_test_reshape_NN.detach().numpy()))
SST = np.sum(np.square(BM_test_true_y.detach().numpy() - np.mean(BM_test_true_y.detach().numpy())))
rSquare_NN = (SST - SSE_NN) / SST


SSE_Biomet = np.sum(np.square(BM_test_true_y.detach().numpy() - BM_Biomet.detach().numpy()))
rSquare_Biomet = (SST - SSE_Biomet) / SST
print("DNN model :", rSquare_NN, "Biomet model : ", rSquare_Biomet)

# In[] # WCA + STOWA prediction and plot
stowawca_test_df = pd.read_csv("STOWA_WCA(645).csv", delimiter=",")
wca_test_df = stowawca_test_df[stowawca_test_df['source']=="WCA"]
wca_test_data = wca_test_df[["pH", "Ca", "DOC", "EU-Chronic", "MLR"]]

wca_test_X, wca_test_y = wca_test_data.iloc[:,:3], wca_test_data.iloc[:,3]
wca_MLR = wca_test_data.iloc[:,4]

wca_test_input_x = torch.Tensor(wca_test_X.values)
wca_test_true_y = torch.Tensor(wca_test_y.values)

wca_test_reshape_NN = torch.reshape(model(wca_test_input_x), (-1,))

plt.rc('figure', dpi=300, figsize=(12, 12))
#plt.figure(figsize=(12,12))
plt.rc('axes', linewidth=2)
plt.rcParams["font.family"] = "Times New Roman"
plotRange = 40

plt.subplot(221)
plt.axis([0, plotRange, 0, plotRange])
#plt.title("Deep learning model")
plt.xlabel("PNEC value (\u03bcg/L)")
plt.ylabel("DNN model Prediction (\u03bcg/L)")
_ = plt.plot(wca_test_true_y.detach().numpy(), wca_test_reshape_NN.detach().numpy(), 'ko', mfc = 'none')
_ = plt.plot(np.arange(0,plotRange + 1,1), np.arange(0,plotRange + 1,1), linewidth = 1)

a = wca_test_reshape_NN.detach().numpy()
a = pd.DataFrame(a)
a.to_csv("wca_test_reshape_NN.csv")
b = wca_test_input_x.detach().numpy()
b = pd.DataFrame(b)
b.to_csv("wca_test_input_x.csv")
c = wca_test_true_y.detach().numpy()
c = pd.DataFrame(c)
c.to_csv("wca_test_true_y.csv")
 
wca_MLR = torch.Tensor(wca_MLR.values)

plt.subplot(222)
plt.axis([0, plotRange, 0, plotRange])
#plt.title("Deep learning model")
plt.xlabel("PNEC value (\u03bcg/L)")
plt.ylabel("Multi-linear regression model Prediction (\u03bcg/L)")
_ = plt.plot(wca_test_true_y.detach().numpy(), wca_MLR.detach().numpy(), 'k^', mfc = 'none')
_ = plt.plot(np.arange(0,plotRange + 1,1), np.arange(0,plotRange + 1,1), linewidth = 1)

SSE_NN = np.sum(np.square(wca_test_true_y.detach().numpy() - wca_test_reshape_NN.detach().numpy()))
SST = np.sum(np.square(wca_test_true_y.detach().numpy() - wca.mean(wca_test_true_y.detach().numpy())))
rSquare_NN = (SST - SSE_NN) / SST


SSE_MLR = np.sum(np.square(wca_test_true_y.detach().numpy() - wca_MLR.detach().numpy()))
rSquare_MLR = (SST - SSE_MLR) / SST
print("DNN model :", rSquare_NN, "MLR model : ", rSquare_MLR)

# In[] STOWA prediction
stowa_test_df = stowawca_test_df[stowawca_test_df['source']=="STOWA"]
stowa_test_data = stowa_test_df[["pH", "Ca", "DOC", "EU-Chronic", "MLR"]]

stowa_test_X, stowa_test_y = stowa_test_data.iloc[:,:3], stowa_test_data.iloc[:,3]
stowa_MLR = stowa_test_data.iloc[:,4]

stowa_test_input_x = torch.Tensor(stowa_test_X.values)
stowa_test_true_y = torch.Tensor(stowa_test_y.values)

stowa_test_reshape_NN = torch.reshape(model(stowa_test_input_x), (-1,))

plt.rc('figure', dpi=300, figsize=(12, 12))
#plt.figure(figsize=(12,12))
plt.rc('axes', linewidth=2)
plt.rcParams["font.family"] = "Times New Roman"
plotRange = 120

plt.subplot(221)
plt.axis([0, plotRange, 0, plotRange])
#plt.title("Deep learning model")
plt.xlabel("PNEC value (\u03bcg/L)")
plt.ylabel("DNN model Prediction (\u03bcg/L)")
_ = plt.plot(stowa_test_true_y.detach().numpy(), stowa_test_reshape_NN.detach().numpy(), 'ko', mfc = 'none')
_ = plt.plot(np.arange(0,plotRange + 1,1), np.arange(0,plotRange + 1,1), linewidth = 1)

a = stowa_test_reshape_NN.detach().numpy()
a = pd.DataFrame(a)
a.to_csv("stowa_test_reshape_NN.csv")
b = stowa_test_input_x.detach().numpy()
b = pd.DataFrame(b)
b.to_csv("stowa_test_input_x.csv")
c = stowa_test_true_y.detach().numpy()
c = pd.DataFrame(c)
c.to_csv("stowa_test_true_y.csv")

stowa_MLR = torch.Tensor(stowa_MLR.values)

plt.subplot(222)
plt.axis([0, plotRange, 0, plotRange])
#plt.title("Deep learning model")
plt.xlabel("PNEC value (\u03bcg/L)")
plt.ylabel("Multi-linear regression model Prediction (\u03bcg/L)")
_ = plt.plot(stowa_test_true_y.detach().numpy(), stowa_MLR.detach().numpy(), 'k^', mfc = 'none')
_ = plt.plot(np.arange(0,plotRange + 1,1), np.arange(0,plotRange + 1,1), linewidth = 1)

SSE_NN = np.sum(np.square(stowa_test_true_y.detach().numpy() - stowa_test_reshape_NN.detach().numpy()))
SST = np.sum(np.square(stowa_test_true_y.detach().numpy() - np.mean(stowa_test_true_y.detach().numpy())))
rSquare_NN = (SST - SSE_NN) / SST


SSE_MLR = np.sum(np.square(stowa_test_true_y.detach().numpy() - stowa_MLR.detach().numpy()))
rSquare_MLR = (SST - SSE_MLR) / SST
print("DNN model :", rSquare_NN, "MLR model : ", rSquare_MLR)


# In[] # Belgium_DB 
BEL_test_data = pd.read_csv("Belgium_DB.csv")
BEL_test_data = BEL_test_data[["pH", "Cond", "DOC", "Cu_HC5", "Biomet"]] 
BEL_test_data['Cond'] = np.log10(BEL_test_data['Cond'])
BEL_test_data['DOC'] = np.log10(BEL_test_data['DOC'])

BEL_test_X, BEL_test_y = BEL_test_data.iloc[:,:3], BEL_test_data.iloc[:,3]
BEL_Biomet = BEL_test_data.iloc[:,4]

BEL_test_input_x = torch.Tensor(BEL_test_X.values)
BEL_test_true_y = torch.Tensor(BEL_test_y.values)

BEL_test_reshape_NN = torch.reshape(model(BEL_test_input_x), (-1,))

plt.rc('figure', dpi=300, figsize=(12, 12))
#plt.figure(figsize=(12,12))
plt.rc('axes', linewidth=2)
plt.rcParams["font.family"] = "Times New Roman"
plotRange = 150

plt.subplot(221)
plt.axis([0, plotRange, 0, plotRange])
#plt.title("Deep learning model")
plt.xlabel("PNEC value (\u03bcg/L)")
plt.ylabel("DNN model Prediction (\u03bcg/L)")
_ = plt.plot(BEL_test_true_y.detach().numpy(), BEL_test_reshape_NN.detach().numpy(), 'ko', mfc = 'none', markerfacecolor = 'r', markeredgecolor = 'k', label = 'DNN result')
_ = plt.plot(np.arange(0,plotRange + 1,1), np.arange(0,plotRange + 1,1), linewidth = 2, color = 'k')
plt.legend()
a = BEL_test_reshape_NN.detach().numpy()
a = pd.DataFrame(a)
a.to_csv("Belgium_DB_reshape_NN.csv")
b = BEL_test_input_x.detach().numpy()
b = pd.DataFrame(b)
b.to_csv("Belgium_DB_input_x.csv")
c = BEL_test_true_y.detach().numpy()
c = pd.DataFrame(c)
c.to_csv("Belgium_DB_true_y.csv")

BM_Biomet = torch.Tensor(BM_Biomet.values)

plt.subplot(222)
plt.axis([0, plotRange, 0, plotRange])
#plt.title("Deep learning model")
plt.xlabel("PNEC value (\u03bcg/L)")
plt.ylabel("Bio-Met model Prediction (\u03bcg/L)")
_ = plt.plot(BEL_test_true_y.detach().numpy(), BEL_Biomet.detach().numpy(), 'k^', mfc = 'none',markerfacecolor = 'b', markeredgecolor = 'k', label = 'Bio-met result')
_ = plt.plot(np.arange(0,plotRange + 1,1), np.arange(0,plotRange + 1,1), linewidth = 2, color = 'k')
plt.legend()
SSE_NN = np.sum(np.square(BEL_test_true_y.detach().numpy() - BEL_test_reshape_NN.detach().numpy()))
SST = np.sum(np.square(BEL_test_true_y.detach().numpy() - np.mean(BEL_test_true_y.detach().numpy())))
rSquare_NN = (SST - SSE_NN) / SST


SSE_Biomet = np.sum(np.square(BEL_test_true_y.detach().numpy() - BEL_Biomet.detach().numpy()))
rSquare_Biomet = (SST - SSE_Biomet) / SST
print("DNN model :", rSquare_NN, "Biomet model : ", rSquare_Biomet)





# In[] # test
# In[] # test
# In[] # test
# In[] # test

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import torch
import torch.nn as nn
import torch.optim as optim
#from sklearn.metrics import mean_absolute_error
#from sklearn.metrics import r2_score

# ====== Prepare Dataset ====== #
data = pd.read_csv("Cu_Simul_DB.csv")
data = data[["pH", "Cond", "DOC", "Cu_HC5"]]

data['DOC'] = np.log10(data['DOC'])
data['Cond'] = np.log10(data['Cond'])
data['Cu_HC5'] = np.log10(data['Cu_HC5'])

index = list(range(len(data)))
random.shuffle(index)
#index_sample = random.sample(index, 10000)
shuffled_data = data.iloc[index,:]
divide_num = round(len(shuffled_data) * 0.7)
trn = shuffled_data.iloc[:divide_num,:] # 70%
val = shuffled_data.iloc[divide_num:,:] # 30%

# ====== Split Dataset into Train, Validation ======#
train_X, train_y = trn.iloc[:,:3], trn.iloc[:,3]
val_X, val_y = val.iloc[:,:3], val.iloc[:,3]



# In[]
class MLPModel(nn.Module): # 원래조건 : (5,20) (20,15), (15,10), (10,1)
    def __init__(self): 
        super(MLPModel, self).__init__()
        self.linear1 = nn.Linear(3,12)
        self.sigmoid1 = nn.Sigmoid()
        self.linear2 = nn.Linear(12,9)
        self.sigmoid2 = nn.Sigmoid()
        self.linear3 = nn.Linear(9,6)
        self.relu3 = nn.ReLU()
        self.linear4 = nn.Linear(6,1)

    def forward(self, x):
    # 인스턴스(샘플) x가 인풋으로 들어왔을 때 모델이 예측하는 y값을 리턴합니다.
        x = self.linear1(x)
        x = self.sigmoid1(x)
        x = self.linear2(x)
        x = self.sigmoid2(x)
        x = self.linear3(x)
        x = self.relu3(x)
        x = self.linear4(x)
        return x
    
reg_loss = nn.MSELoss()

model = MLPModel() # Model을 생성해줍니다.
print('{} parameters'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))

lr = 0.005 # Learning Rate를 하나 정해줍니다.
optimizer = optim.Adamax(model.parameters(), lr=lr) # Optimizer를 생성해줍니다.

list_epoch = [] 
list_train_loss = []
list_val_loss = []
list_mae = []
list_mae_epoch = []

# In[]
epoch = 150000 # 학습 횟수(epoch)을 지정해줍시다.

for i in range(epoch):
    
    # ====== Train ====== #
    model.train() # model을 train 모드로 세팅합니다. 반대로 향후 모델을 평가할 때는 eval() 모드로 변경할 겁니다 (나중 실습에서 쓸 겁니다)
    optimizer.zero_grad() # optimizer에 남아있을 수도 있는 잔여 그라디언트를 0으로 다 초기화해줍니다.
    
    input_x = torch.Tensor(train_X.values)
    true_y = torch.Tensor(train_y.values)
    pred_y = model(input_x)
    #print(input_x.shape, true_y.shape, pred_y.shape) # 각 인풋과 아웃풋의 차원을 체크해봅니다.
    
 #loss = reg_loss(pred_y.squeeze(), true_y)
    loss = reg_loss((pred_y.squeeze() + 1), (true_y + 1)) # msle 로그MSE 최소화
    loss.backward() # backward()를 통해서 그라디언트를 구해줍니다.
    optimizer.step() # step()을 통해서 그라디언틀르 바탕으로 파라미터를 업데이트 해줍니다.
    list_epoch.append(i)
    list_train_loss.append(loss.detach().numpy())
    
    
    # ====== Validation ====== #
    model.eval()
    optimizer.zero_grad()
    input_x = torch.Tensor(val_X.values)
    true_y = torch.Tensor(val_y.values)
    pred_y = model(input_x)
    loss = reg_loss(pred_y.squeeze(), true_y)
    list_val_loss.append(loss.detach().numpy())
    

    # ====== Evaluation ======= #
    if i % 200 == 0: # 200회의 학습마다 실제 데이터 분포와 모델이 예측한 분포를 그려봅니다.
        
        # ====== Calculate MAE ====== #
        #model.eval()
        #optimizer.zero_grad()
        #input_x = torch.Tensor(test_X.values)
        #true_y = torch.Tensor(test_y.values)
        #pred_y = model(input_x).detach().numpy() 
        #mae = mean_absolute_error(true_y, pred_y) # sklearn 쪽 함수들은 true_y 가 먼저, pred_y가 나중에 인자로 들어가는 것에 주의합시다
        #list_mae.append(mae)
        #list_mae_epoch.append(i)
        
        print(i, loss)

torch.save(model.state_dict(), 'C:\\Users\\juvox\\model\\cond_training_model.pth') # 모델 저장

model.load_state_dict(torch.load('C:\\Users\\juvox\\model\\cond_training_model.pth')) # 모델 불러오기




# In[] # training data prediction value output
data_X, data_y = data.iloc[:,:3], data.iloc[:,3]

data_input_x = torch.Tensor(data_X.values)
data_true_y = torch.Tensor(data_y.values)

data_reshape_NN = torch.reshape(model(data_input_x), (-1,))

plt.rc('figure', dpi=300, figsize=(12, 12))
#plt.figure(figsize=(12,12))
plt.rc('axes', linewidth=2)
plt.rcParams["font.family"] = "Times New Roman"
plotRange = 300

plt.subplot(221)
plt.axis([0, plotRange, 0, plotRange])
#plt.title("Deep learning model")
plt.xlabel("PNEC value (\u03bcg/L)")
plt.ylabel("DNN model Prediction (\u03bcg/L)")
_ = plt.plot(data_true_y.detach().numpy(), data_reshape_NN.detach().numpy(), 'ko', mfc = 'none')
_ = plt.plot(np.arange(0,plotRange + 1,1), np.arange(0,plotRange + 1,1), linewidth = 1)
#_ = plt.plot(np.arange(0,5,1), np.arange(0,5,1), linewidth = 1)

a = data_reshape_NN.detach().numpy()
a = pd.DataFrame(a)
b = 10 ** a
b.to_csv("data_reshape_NN(log).csv")


# In[] # kr test set 764개 그래프
kr_test_data = pd.read_csv("kr_test(log).csv")
kr_test_data = kr_test_data[["pH", "Cond", "DOC", "EU_Chronic HC5", "MLR"]]
kr_test_data['Cond'] = np.log10(kr_test_data['Cond'])
kr_test_data['DOC'] = np.log10(kr_test_data['DOC'])

kr_test_X, kr_test_y = kr_test_data.iloc[:,:3], kr_test_data.iloc[:,3]
kr_MLR = kr_test_data.iloc[:,4]

kr_test_input_x = torch.Tensor(kr_test_X.values)
kr_test_true_y = torch.Tensor(kr_test_y.values)

kr_test_reshape_NN = torch.reshape(model(kr_test_input_x), (-1,))

plt.rc('figure', dpi=300, figsize=(12, 12))
#plt.figure(figsize=(12,12))
plt.rc('axes', linewidth=2)
plt.rcParams["font.family"] = "Times New Roman"
plotRange = 150

plt.subplot(221)
plt.axis([0, plotRange, 0, plotRange])
#plt.title("Deep learning model")
plt.xlabel("PNEC value (\u03bcg/L)")
plt.ylabel("DNN model Prediction (\u03bcg/L)")
_ = plt.plot(kr_test_true_y.detach().numpy(), kr_test_reshape_NN.detach().numpy(), 'ko', mfc = 'none', markerfacecolor = 'r', markeredgecolor = 'k', label = 'DNN result')
_ = plt.plot(np.arange(0,plotRange + 1,1), np.arange(0,plotRange + 1,1), linewidth = 2, color = 'k')
plt.legend()
a = kr_test_reshape_NN.detach().numpy()
a = pd.DataFrame(a)
b = 10 ** a
b.to_csv("kr_test_reshape_NN(log).csv")



# In[] # Belgium_DB 
BEL_test_data = pd.read_csv("Belgium_DB.csv")
BEL_test_data = BEL_test_data[["pH", "Cond", "DOC", "Cu_HC5", "Biomet"]] 
BEL_test_data['Cond'] = np.log10(BEL_test_data['Cond'])
BEL_test_data['DOC'] = np.log10(BEL_test_data['DOC'])

BEL_test_X, BEL_test_y = BEL_test_data.iloc[:,:3], BEL_test_data.iloc[:,3]
BEL_Biomet = BEL_test_data.iloc[:,4]

BEL_test_input_x = torch.Tensor(BEL_test_X.values)
BEL_test_true_y = torch.Tensor(BEL_test_y.values)

BEL_test_reshape_NN = torch.reshape(model(BEL_test_input_x), (-1,))

plt.rc('figure', dpi=300, figsize=(12, 12))
#plt.figure(figsize=(12,12))
plt.rc('axes', linewidth=2)
plt.rcParams["font.family"] = "Times New Roman"
plotRange = 150

plt.subplot(221)
plt.axis([0, plotRange, 0, plotRange])
#plt.title("Deep learning model")
plt.xlabel("PNEC value (\u03bcg/L)")
plt.ylabel("DNN model Prediction (\u03bcg/L)")
_ = plt.plot(BEL_test_true_y.detach().numpy(), BEL_test_reshape_NN.detach().numpy(), 'ko', mfc = 'none', markerfacecolor = 'r', markeredgecolor = 'k', label = 'DNN result')
_ = plt.plot(np.arange(0,plotRange + 1,1), np.arange(0,plotRange + 1,1), linewidth = 2, color = 'k')
plt.legend()
a = BEL_test_reshape_NN.detach().numpy()
a = pd.DataFrame(a)
b = 10 ** a
b.to_csv("Belgium_DB_reshape_NN.csv")


# In[] # US + SW prediction and plot
ussw_test_df = pd.read_csv("US_SW(536).csv", delimiter=",")
us_test_df = ussw_test_df[(ussw_test_df['source']=="US EPA") | (ussw_test_df['source']== "Oregon") | (ussw_test_df['source']== "add_Oregon")]
us_test_data = us_test_df[["pH", "Cond", "DOC", "EU-Chronic", "MLR"]]
us_test_data['Cond'] = np.log10(us_test_data['Cond'])
us_test_data['DOC'] = np.log10(us_test_data['DOC'])

us_test_X, us_test_y = us_test_data.iloc[:,:3], us_test_data.iloc[:,3]
us_MLR = us_test_data.iloc[:,4]

us_test_input_x = torch.Tensor(us_test_X.values)
us_test_true_y = torch.Tensor(us_test_y.values)

us_test_reshape_NN = torch.reshape(model(us_test_input_x), (-1,))

plt.rc('figure', dpi=300, figsize=(12, 12))
#plt.figure(figsize=(12,12))
plt.rc('axes', linewidth=2)
plt.rcParams["font.family"] = "Times New Roman"
plotRange = 40

plt.subplot(221)
plt.axis([0, plotRange, 0, plotRange])
#plt.title("Deep learning model")
plt.xlabel("PNEC value (\u03bcg/L)")
plt.ylabel("DNN model Prediction (\u03bcg/L)")
_ = plt.plot(us_test_true_y.detach().numpy(), us_test_reshape_NN.detach().numpy(), 'ko', mfc = 'none')
_ = plt.plot(np.arange(0,plotRange + 1,1), np.arange(0,plotRange + 1,1), linewidth = 1)

a = us_test_reshape_NN.detach().numpy()
a = pd.DataFrame(a)
b = 10 ** a
b.to_csv("us_test_reshape_NN(log).csv")

# In[] sw prediction
sw_test_df = ussw_test_df[ussw_test_df['source']=="SW"]
sw_test_data = sw_test_df[["pH", "Cond", "DOC", "EU-Chronic", "MLR"]]
sw_test_data['Cond'] = np.log10(sw_test_data['Cond'])
sw_test_data['DOC'] = np.log10(sw_test_data['DOC'])

sw_test_X, sw_test_y = sw_test_data.iloc[:,:3], sw_test_data.iloc[:,3]
sw_MLR = sw_test_data.iloc[:,4]

sw_test_input_x = torch.Tensor(sw_test_X.values)
sw_test_true_y = torch.Tensor(sw_test_y.values)

sw_test_reshape_NN = torch.reshape(model(sw_test_input_x), (-1,))

plt.rc('figure', dpi=300, figsize=(12, 12))
#plt.figure(figsize=(12,12))
plt.rc('axes', linewidth=2)
plt.rcParams["font.family"] = "Times New Roman"
plotRange = 120

plt.subplot(221)
plt.axis([0, plotRange, 0, plotRange])
#plt.title("Deep learning model")
plt.xlabel("PNEC value (\u03bcg/L)")
plt.ylabel("DNN model Prediction (\u03bcg/L)")
_ = plt.plot(sw_test_true_y.detach().numpy(), sw_test_reshape_NN.detach().numpy(), 'ko', mfc = 'none')
_ = plt.plot(np.arange(0,plotRange + 1,1), np.arange(0,plotRange + 1,1), linewidth = 1)

a = sw_test_reshape_NN.detach().numpy()
a = pd.DataFrame(a)
b = 10 ** a
b.to_csv("sw_test_reshape_NN(log).csv")



# In[] # Ca_Ca_Ca_Ca_Ca_Ca_Ca_Ca_Ca_test
# In[] # test
# In[] # test
# In[] # test

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import torch
import torch.nn as nn
import torch.optim as optim
#from sklearn.metrics import mean_absolute_error
#from sklearn.metrics import r2_score

# ====== Prepare Dataset ====== #
data = pd.read_csv("Cu_Simul_DB.csv")
data = data[["pH", "Ca", "DOC", "Cu_HC5"]]

data['DOC'] = np.log10(data['DOC'])
data['Ca'] = np.log10(data['Ca'])


index = list(range(len(data)))
random.shuffle(index)
#index_sample = random.sample(index, 10000)
shuffled_data = data.iloc[index,:]
divide_num = round(len(shuffled_data) * 0.7)
trn = shuffled_data.iloc[:divide_num,:] # 70%
val = shuffled_data.iloc[divide_num:,:] # 30%

# ====== Split Dataset into Train, Validation ======#
train_X, train_y = trn.iloc[:,:3], trn.iloc[:,3]
val_X, val_y = val.iloc[:,:3], val.iloc[:,3]


# In[]
class MLPModel(nn.Module): # 원래조건 : (5,20) (20,15), (15,10), (10,1)
    def __init__(self): 
        super(MLPModel, self).__init__()
        self.linear1 = nn.Linear(3,12)
        self.sigmoid1 = nn.Sigmoid()
        self.linear2 = nn.Linear(12,9)
        self.sigmoid2 = nn.Sigmoid()
        self.linear3 = nn.Linear(9,6)
        self.relu3 = nn.ReLU()
        self.linear4 = nn.Linear(6,1)

    def forward(self, x):
    # 인스턴스(샘플) x가 인풋으로 들어왔을 때 모델이 예측하는 y값을 리턴합니다.
        x = self.linear1(x)
        x = self.sigmoid1(x)
        x = self.linear2(x)
        x = self.sigmoid2(x)
        x = self.linear3(x)
        x = self.relu3(x)
        x = self.linear4(x)
        return x
    
reg_loss = nn.MSELoss()

model = MLPModel() # Model을 생성해줍니다.
print('{} parameters'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))

lr = 0.005 # Learning Rate를 하나 정해줍니다.
optimizer = optim.Adamax(model.parameters(), lr=lr) # Optimizer를 생성해줍니다.

list_epoch = [] 
list_train_loss = []
list_val_loss = []
list_mae = []
list_mae_epoch = []

# In[]
epoch = 150000 # 학습 횟수(epoch)을 지정해줍시다.

for i in range(epoch):
    
    # ====== Train ====== #
    model.train() # model을 train 모드로 세팅합니다. 반대로 향후 모델을 평가할 때는 eval() 모드로 변경할 겁니다 (나중 실습에서 쓸 겁니다)
    optimizer.zero_grad() # optimizer에 남아있을 수도 있는 잔여 그라디언트를 0으로 다 초기화해줍니다.
    
    input_x = torch.Tensor(train_X.values)
    true_y = torch.Tensor(train_y.values)
    pred_y = model(input_x)
    #print(input_x.shape, true_y.shape, pred_y.shape) # 각 인풋과 아웃풋의 차원을 체크해봅니다.
    
    #loss = reg_loss(pred_y.squeeze(), true_y)
    loss = reg_loss(torch.log10(pred_y.squeeze() + 1), torch.log10(true_y + 1)) # msle 로그MSE 최소화
    loss.backward() # backward()를 통해서 그라디언트를 구해줍니다.
    optimizer.step() # step()을 통해서 그라디언틀르 바탕으로 파라미터를 업데이트 해줍니다.
    list_epoch.append(i)
    list_train_loss.append(loss.detach().numpy())
    
    
    # ====== Validation ====== #
    model.eval()
    optimizer.zero_grad()
    input_x = torch.Tensor(val_X.values)
    true_y = torch.Tensor(val_y.values)
    pred_y = model(input_x)
    loss = reg_loss(pred_y.squeeze(), true_y)
    list_val_loss.append(loss.detach().numpy())
    

    # ====== Evaluation ======= #
    if i % 200 == 0: # 200회의 학습마다 실제 데이터 분포와 모델이 예측한 분포를 그려봅니다.
        
        # ====== Calculate MAE ====== #
        #model.eval()
        #optimizer.zero_grad()
        #input_x = torch.Tensor(test_X.values)
        #true_y = torch.Tensor(test_y.values)
        #pred_y = model(input_x).detach().numpy() 
        #mae = mean_absolute_error(true_y, pred_y) # sklearn 쪽 함수들은 true_y 가 먼저, pred_y가 나중에 인자로 들어가는 것에 주의합시다
        #list_mae.append(mae)
        #list_mae_epoch.append(i)
        
        print(i, loss)

torch.save(model.state_dict(), 'C:\\Users\\juvox\\model\\Ca_training_model.pth') # 모델 저장

model.load_state_dict(torch.load('C:\\Users\\juvox\\model\\Ca_training_model.pth')) # 모델 불러오기


# In[] # training data prediction value output
data_X, data_y = data.iloc[:,:3], data.iloc[:,3]

data_input_x = torch.Tensor(data_X.values)
data_true_y = torch.Tensor(data_y.values)

data_reshape_NN = torch.reshape(model(data_input_x), (-1,))

plt.rc('figure', dpi=300, figsize=(12, 12))
#plt.figure(figsize=(12,12))
plt.rc('axes', linewidth=2)
plt.rcParams["font.family"] = "Times New Roman"
plotRange = 300

plt.subplot(221)
plt.axis([0, plotRange, 0, plotRange])
#plt.title("Deep learning model")
plt.xlabel("PNEC value (\u03bcg/L)")
plt.ylabel("DNN model Prediction (\u03bcg/L)")
_ = plt.plot(data_true_y.detach().numpy(), data_reshape_NN.detach().numpy(), 'ko', mfc = 'none')
_ = plt.plot(np.arange(0,plotRange + 1,1), np.arange(0,plotRange + 1,1), linewidth = 1)
#_ = plt.plot(np.arange(0,5,1), np.arange(0,5,1), linewidth = 1)

a = data_reshape_NN.detach().numpy()
a = pd.DataFrame(a)
#b = 10 ** a
a.to_csv("data_reshape_NN(log).csv")


# In[] # kr test set 764개 그래프
kr_test_data = pd.read_csv("kr_test(log).csv")
kr_test_data = kr_test_data[["pH", "Ca", "DOC", "EU_Chronic HC5", "MLR"]]
kr_test_data['Ca'] = np.log10(kr_test_data['Ca'])
kr_test_data['DOC'] = np.log10(kr_test_data['DOC'])

kr_test_X, kr_test_y = kr_test_data.iloc[:,:3], kr_test_data.iloc[:,3]
kr_MLR = kr_test_data.iloc[:,4]

kr_test_input_x = torch.Tensor(kr_test_X.values)
kr_test_true_y = torch.Tensor(kr_test_y.values)

kr_test_reshape_NN = torch.reshape(model(kr_test_input_x), (-1,))

plt.rc('figure', dpi=300, figsize=(12, 12))
#plt.figure(figsize=(12,12))
plt.rc('axes', linewidth=2)
plt.rcParams["font.family"] = "Times New Roman"
plotRange = 150

plt.subplot(221)
plt.axis([0, plotRange, 0, plotRange])
#plt.title("Deep learning model")
plt.xlabel("PNEC value (\u03bcg/L)")
plt.ylabel("DNN model Prediction (\u03bcg/L)")
_ = plt.plot(kr_test_true_y.detach().numpy(), kr_test_reshape_NN.detach().numpy(), 'ko', mfc = 'none', markerfacecolor = 'r', markeredgecolor = 'k', label = 'DNN result')
_ = plt.plot(np.arange(0,plotRange + 1,1), np.arange(0,plotRange + 1,1), linewidth = 2, color = 'k')
plt.legend()
a = kr_test_reshape_NN.detach().numpy()
a = pd.DataFrame(a)
#b = 10 ** a
a.to_csv("kr_test_reshape_NN(log).csv")



# In[] # Belgium_DB 
BEL_test_data = pd.read_csv("Belgium_DB.csv")
BEL_test_data = BEL_test_data[["pH", "Cond", "DOC", "Cu_HC5", "Biomet"]] 
BEL_test_data['Cond'] = np.log10(BEL_test_data['Cond'])
BEL_test_data['DOC'] = np.log10(BEL_test_data['DOC'])

BEL_test_X, BEL_test_y = BEL_test_data.iloc[:,:3], BEL_test_data.iloc[:,3]
BEL_Biomet = BEL_test_data.iloc[:,4]

BEL_test_input_x = torch.Tensor(BEL_test_X.values)
BEL_test_true_y = torch.Tensor(BEL_test_y.values)

BEL_test_reshape_NN = torch.reshape(model(BEL_test_input_x), (-1,))

plt.rc('figure', dpi=300, figsize=(12, 12))
#plt.figure(figsize=(12,12))
plt.rc('axes', linewidth=2)
plt.rcParams["font.family"] = "Times New Roman"
plotRange = 150

plt.subplot(221)
plt.axis([0, plotRange, 0, plotRange])
#plt.title("Deep learning model")
plt.xlabel("PNEC value (\u03bcg/L)")
plt.ylabel("DNN model Prediction (\u03bcg/L)")
_ = plt.plot(BEL_test_true_y.detach().numpy(), BEL_test_reshape_NN.detach().numpy(), 'ko', mfc = 'none', markerfacecolor = 'r', markeredgecolor = 'k', label = 'DNN result')
_ = plt.plot(np.arange(0,plotRange + 1,1), np.arange(0,plotRange + 1,1), linewidth = 2, color = 'k')
plt.legend()
a = BEL_test_reshape_NN.detach().numpy()
a = pd.DataFrame(a)
#b = 10 ** a
a.to_csv("Belgium_DB_reshape_NN.csv")


# In[] # US + SW prediction and plot
ussw_test_df = pd.read_csv("US_SW(536).csv", delimiter=",")
us_test_df = ussw_test_df[(ussw_test_df['source']=="US EPA") | (ussw_test_df['source']== "Oregon") | (ussw_test_df['source']== "add_Oregon")]
us_test_data = us_test_df[["pH", "Cond", "DOC", "EU-Chronic", "MLR"]]
us_test_data['Cond'] = np.log10(us_test_data['Cond'])
us_test_data['DOC'] = np.log10(us_test_data['DOC'])

us_test_X, us_test_y = us_test_data.iloc[:,:3], us_test_data.iloc[:,3]
us_MLR = us_test_data.iloc[:,4]

us_test_input_x = torch.Tensor(us_test_X.values)
us_test_true_y = torch.Tensor(us_test_y.values)

us_test_reshape_NN = torch.reshape(model(us_test_input_x), (-1,))

plt.rc('figure', dpi=300, figsize=(12, 12))
#plt.figure(figsize=(12,12))
plt.rc('axes', linewidth=2)
plt.rcParams["font.family"] = "Times New Roman"
plotRange = 40

plt.subplot(221)
plt.axis([0, plotRange, 0, plotRange])
#plt.title("Deep learning model")
plt.xlabel("PNEC value (\u03bcg/L)")
plt.ylabel("DNN model Prediction (\u03bcg/L)")
_ = plt.plot(us_test_true_y.detach().numpy(), us_test_reshape_NN.detach().numpy(), 'ko', mfc = 'none')
_ = plt.plot(np.arange(0,plotRange + 1,1), np.arange(0,plotRange + 1,1), linewidth = 1)

a = us_test_reshape_NN.detach().numpy()
a = pd.DataFrame(a)
#b = 10 ** a
a.to_csv("us_test_reshape_NN(log).csv")

# In[] sw prediction
sw_test_df = ussw_test_df[ussw_test_df['source']=="SW"]
sw_test_data = sw_test_df[["pH", "Cond", "DOC", "EU-Chronic", "MLR"]]
sw_test_data['Cond'] = np.log10(sw_test_data['Cond'])
sw_test_data['DOC'] = np.log10(sw_test_data['DOC'])

sw_test_X, sw_test_y = sw_test_data.iloc[:,:3], sw_test_data.iloc[:,3]
sw_MLR = sw_test_data.iloc[:,4]

sw_test_input_x = torch.Tensor(sw_test_X.values)
sw_test_true_y = torch.Tensor(sw_test_y.values)

sw_test_reshape_NN = torch.reshape(model(sw_test_input_x), (-1,))

plt.rc('figure', dpi=300, figsize=(12, 12))
#plt.figure(figsize=(12,12))
plt.rc('axes', linewidth=2)
plt.rcParams["font.family"] = "Times New Roman"
plotRange = 120

plt.subplot(221)
plt.axis([0, plotRange, 0, plotRange])
#plt.title("Deep learning model")
plt.xlabel("PNEC value (\u03bcg/L)")
plt.ylabel("DNN model Prediction (\u03bcg/L)")
_ = plt.plot(sw_test_true_y.detach().numpy(), sw_test_reshape_NN.detach().numpy(), 'ko', mfc = 'none')
_ = plt.plot(np.arange(0,plotRange + 1,1), np.arange(0,plotRange + 1,1), linewidth = 1)

a = sw_test_reshape_NN.detach().numpy()
a = pd.DataFrame(a)
#b = 10 ** a
a.to_csv("sw_test_reshape_NN(log).csv")