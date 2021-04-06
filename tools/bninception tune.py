import torch
state_dict = torch.load('bn_inception-9f5701afb96c8044.pth')
for name, weights in state_dict.items():
    print(name, weights.size())  #可以查看模型中的模型名字和权重维度
    if len(weights.size()) == 2: #判断需要修改维度的条件
        state_dict[name] = weights.squeeze(0)  #去掉维度0，把(1,128)转为(128)
        print(name,weights.squeeze(0).size()) #查看转化后的模型名字和权重维度
torch.save(state_dict, 'bn_inception-9f5701afb96c8044.pth')
