
#%%
import numpy as np 
import matplotlib.pyplot as plt

# make random data 
# data = np.random.randint(10, size=(2,20))
# intented_err = np.random.randn(20)
data = [ 
    [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19], 
    [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]
]
data[0] += np.random.randn(20)
data[1] += np.random.randn(20)
plt.scatter(data[0], data[1])


#%% make default regression line
def reg_line_plot(w=1,b=0):
    fig = plt.figure()
    ax = plt.axes()
    x = np.linspace(0, 20, 20)
    ax.scatter(data[0], data[1])
    ax.plot(x, w*x + b, linestyle='solid')

reg_line_plot(2,-4)



#%% 
def gap(w,n):
    x = data[0][n]
    y = data[1][n]
    return y - int(w*x + b)

def loss(w):
    loses = 0;
    for i in range(len(data[0])):
        loses += gap(w,i)
    return loses/len(data[0])


#%% calculate loss and draw graph
w = -1
print("w:", w, " ", "loss:",loss(w))
reg_line_plot(w)

#%%
w = 0
print("w:", w, " ", "loss:",loss(w))
reg_line_plot(w)
w = 1

#%%
print("w:", w, " ", "loss:",loss(w))
reg_line_plot(w)



#%% draw loss funtion
loss_plots = []
for w in np.arange(-10, 10, 0.1):
    loss_plots.append(
        [w, np.abs(loss(w)) ] 
    )

# np.transpose(loss_plots)[1]
plt.plot(np.transpose(loss_plots)[0], np.transpose(loss_plots)[1])
# np.transpose(loss_plots)
# plt.plot(np.transpose(loss_plots))

