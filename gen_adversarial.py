import torch
import torchvision
from torchvision.models import alexnet
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.autograd import Variable
import torch.nn as nn
from PIL import Image

save_iter = True

def adversarial(model, im, targ_class, num_iter=50,loss=nn.CrossEntropyLoss()):
    targ = torch.LongTensor(1).zero_() 
    targ[0] = targ_class
    targ = Variable(targ)
    optimizer = torch.optim.SGD([im],lr=0.01)
    sm = nn.Softmax()
    for i in range(num_iter):
        res = model(im)
        if i ==0:
            print res
        error = loss(res,targ)
        error.backward()
        prob = sm(res)
        print i, prob[0,targ_class].data
        optimizer.step()
        if save_iter:
            im_clone = im.clone()
            im_clone = im_clone.data
            unnormalize(im_clone)
            save_image(im_clone.squeeze(),'out_iter_'+str(i)+'.png')
    return im


model = alexnet(pretrained=True)
model.eval() #so dropout is fixed

mean_vec=[0.485, 0.456, 0.406]
std_vec = [0.229, 0.224, 0.225]

normalize = transforms.Normalize(mean=mean_vec,std=std_vec)
unmean = [-1*x/y for x,y in zip(mean_vec, std_vec)]
unstd  = [1/x for x in std_vec]

unnormalize = transforms.Normalize(mean=unmean, std=unstd)
pre_transform=transforms.Compose([
                transforms.Scale(256),
                transforms.CenterCrop(224),
                torchvision.transforms.ToTensor(),
                normalize])

filepath = 'data/training/n01645776/n01645776_10130.JPEG'

temp = Image.open(filepath)
im = temp.copy()
temp.close()
im = pre_transform(im)
v = Variable(im.unsqueeze(0),requires_grad=True)
adv_im = adversarial(model,v,0)
adv_im = adv_im.data
unnormalize(adv_im)
save_image(adv_im.squeeze(),'adv.png')
