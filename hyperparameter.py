input_unit=1
hidden_layar=1
hidden_layar_unit=10 
output_unit=1

train_step=200 
lr=0.2
name="model"

# Demo
import torch
import matplotlib.pyplot as plt

# For Regression Demo
x=torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1) 
y = x.pow(2) + 0.2*torch.rand(x.size())
loss_func = torch.nn.MSELoss()  # this is for regression mean squared loss
plt.scatter(x.data.numpy(), y.data.numpy())
plt.show()
testx=torch.unsqueeze(torch.linspace(-1, 1, 1), dim=1) 
#For Classifier Demo
# n_data = torch.ones(100, 2)
# x0 = torch.normal(2*n_data, 1)      # class0 x data (tensor), shape=(100, 2)
# y0 = torch.zeros(100)               # class0 y data (tensor), shape=(100, 1)
# x1 = torch.normal(-2*n_data, 1)     # class1 x data (tensor), shape=(100, 2)
# y1 = torch.ones(100)                # class1 y data (tensor), shape=(100, 1)
# x = torch.cat((x0, x1), 0).type(torch.FloatTensor)  # shape (200, 2) FloatTensor = 32-bit floating
# y = torch.cat((y0, y1), ).type(torch.LongTensor)    # shape (200,) LongTensor = 64-bit integer
# plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=y.data.numpy(), s=100, lw=0, cmap='RdYlGn')
# plt.show()
# loss_func = torch.nn.CrossEntropyLoss()

# print(x[0]) 
