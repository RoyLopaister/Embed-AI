import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import hyperparameter as hp
# torch.manual_seed(1)    # reproducible

x = hp.x  # x data (tensor), shape=(100, 1)
y = hp.y  # noisy y data (tensor), shape=(100, 1)


class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)   # hidden layer
        self.predict = torch.nn.Linear(n_hidden, n_output)   # output layer

    def forward(self, x):
        x = F.relu(self.hidden(x))      # activation function for hidden layer
        x = self.predict(x)             # linear output
        return x

net = Net(n_feature=hp.input_unit, n_hidden=hp.hidden_layar_unit, n_output=hp.output_unit)     # define the network
print(net)  # net architecture

optimizer = torch.optim.SGD(net.parameters(), lr=hp.lr)

loss_func = hp.loss_func
plt.ion()   # something about plotting

for t in range(hp.train_step):
    out = net(x)     # input x and predict based on x

    loss = loss_func(out, y)     # must be (1. nn output, 2. target)
    optimizer.zero_grad()   # clear gradients for next train
    loss.backward()         # backpropagation, compute gradients
    optimizer.step()        # apply gradients
    print()
    # if t % 2 == 0:
    #     # plot and show learning process
    #     plt.cla()
    #     prediction = torch.max(out, 1)[1]
    #     pred_y = prediction.data.numpy()
    #     target_y = y.data.numpy()
    #     plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=pred_y, s=100, lw=0, cmap='RdYlGn')
    #     accuracy = float((pred_y == target_y).astype(int).sum()) / float(target_y.size)
    #     plt.text(1.5, -4, 'Accuracy=%.2f' % accuracy, fontdict={'size': 20, 'color':  'red'})
    #     plt.pause(0.1)
    if t % 5 == 0:
        # plot and show learning process
        plt.cla()
        plt.scatter(x.data.numpy(), y.data.numpy())
        plt.plot(x.data.numpy(), out.data.numpy(), 'r-', lw=5)
        plt.text(0.5, 0, 'Loss=%.4f' % loss.data.numpy(), fontdict={'size': 20, 'color':  'red'})
        plt.pause(0.1)

plt.ioff()
plt.show()

for parameters in net.parameters():
  print(parameters)
     
index2=hp.hidden_layar_unit*(hp.hidden_layar_unit*hp.hidden_layar-hp.hidden_layar_unit+hp.input_unit+hp.output_unit)/(hp.hidden_layar+1)
if int(index2)!=index2:
  raise NameError('Check your parameter is right or not')
weight="weight["+str(hp.hidden_layar+1)+"]"+"["+str(int(index2))+"]"+"={\n" 
bias="bias["+str(hp.hidden_layar*hp.hidden_layar_unit+hp.output_unit)+"]={\n" 
count=0 


print("-------------------------------")

for parameters in net.parameters(): 
  if count%2==0:
    #weight
    w=parameters.detach().numpy()
    for i in range(len(w)):
        if i==0:
            weight=weight+"{"
        try:
            for j in range(len(w[i])):
                weights=w[i][j]
                weight=weight+str(weights)
                if j != len(w[i])-1:
                    weight=weight+","
        except IndexError:
            continue
        if i != len(w)-1:
            weight=weight+","
    weight=weight+"},\n"
  else:
    b=parameters.detach().numpy()
    for i in range(len(b)):
      bias=bias+str(b[i])
      bias=bias+","
  count+=1

print("-------------------------------")
weight=weight[:-2]+"\n};\n"
bias=bias[:-2]+"\n};"
print(weight)
print(bias)

from  datetime import datetime, timedelta
now_time = datetime.now()
new_time = now_time.strftime('%Y:%m:%d %H-%M-%S')
import os
path = "Model/"+new_time
if not os.path.isdir(path):
    os.mkdir(path)
f = open(path+"/"+hp.name+".h", "w")
topology="int topology["+str(hp.hidden_layar+2)+"]={"+str(hp.input_unit)+","
for i in range(hp.hidden_layar):
    topology=topology+str(hp.hidden_layar_unit)+","
topology=topology+str(hp.output_unit)+"};\nint outnode="+str(hp.input_unit+hp.hidden_layar_unit*hp.hidden_layar+hp.output_unit)+";\nfloat ReLu(float in){\n\treturn (in<0) ? 0:in;\n}\n"
f.write("/*  "+new_time+"   Fully connect neural network"+"  */\n\n"+topology+"\n\nfloat "+weight+"\nfloat "+bias)
f.close()


print(hp.testx)
print(net(hp.testx))
