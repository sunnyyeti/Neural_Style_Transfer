#!/usr/bin/python  
#-*- coding:utf-8 -*-  

############################  
#File Name: NeuralStyle.py  
#Author: HaoZhang  
#Mail: sunnyyetizh@gmail.com  
#Created Time: 2018-01-17
############################  

from __future__ import print_function
import argparse

def required_length(nmin,nmax):
    class RequiredLength(argparse.Action):
        def __call__(self, parser, args, values, option_string=None):
            if not nmin<=len(values)<=nmax:
                msg='argument "{f}" requires between {nmin} and {nmax} arguments'.format(
                    f=self.dest,nmin=nmin,nmax=nmax)
                raise argparse.ArgumentTypeError(msg)
            setattr(args, self.dest, values)
    return RequiredLength

parser = argparse.ArgumentParser(description="Generate an image with content as the 'content' image and style as the 'style' image(s) you provide.")
parser.add_argument("--sty_imgs","-s",nargs="+",required=True, 
                    help="File path(es) to the style image(s). If there are more than one style image, please separate them by space.")
parser.add_argument("--style_blend_weights",'-sbw',nargs='+',type=float,
                    help="Weights of different style to blend. This should have the same length as option '--sty_img/-s'. Please separate weights by space. Equal weights are assigned to different styles if it is left default.")
parser.add_argument("--con_img","-c",required=True,
                    help="File path to the content image.")
parser.add_argument("--out","-o",required=True,
                    help="File path to the output image.")
parser.add_argument("--out_size",'-os',nargs="+",action=required_length(1,2),type=int,
                    help="Size of the output image. By default, the output image has the same size as content image. You can specify the height and width by separating them with space like 'h w'. If you only specify an Int, then the size of the output is determined by matching the smaller edge of the content image to this number while keeping the ratio.")
parser.add_argument("--content_layers",'-cls',nargs="+",default=["relu_4"],
                    help="Layers used to reconstruct content. Please only use 'relu' and 'conv' in format 'relu_i' and 'conv_i' with 1<=i<=16. If there are more than one layer, specify them by space. Default is ['relu_4'].")
parser.add_argument("--style_layers",'-sls',nargs="+",default=['relu_1', 'relu_2', 'relu_3', 'relu_4', 'relu_5'],
                    help="Layers used to reconstruct style. Please only use 'relu' and 'conv' in format 'relu_i' and 'conv_i' with 1<=i<=16. If there are more than one layer, specify them by space. Default is ['relu_1', 'relu_2', 'relu_3', 'relu_4', 'relu_5'].")
parser.add_argument("--content_weight","-cw",type=float,default=0.5,
                    help="Weight of the reconstruction of the content. Default is 0.5.")
parser.add_argument("--style_weight","-sw",type=float,default=5e3,
                    help="Weight of the reconstruction of the style. Default is 5000.0.")
parser.add_argument("--optim",default="lbfgs",choices=['adam','lbfgs'],
                    help="Specify the optimization algorithm. Please choose from 'adam' and 'lbfgs'. Default is 'lbfgs'.")
parser.add_argument("--learning_rate",'-lr',default=0.7,type=float,
                    help="Learning rate for optimization algorithm. Default is 0.7.")
parser.add_argument("--num_iter",'-ni',default=20,type=int,
                    help="Number of the iterations. Default is 20.")
parser.add_argument("--use_cuda","-uc",action="store_true",
                    help="Switch to use CUDA to accelerate computing on GPU.")
parser.add_argument("--pooling","-p",default="ave",choices=["max","ave"],
                    help="Type of pooling layer. Choose from 'max' and 'ave'. Default is 'ave'.")
parser.add_argument("--init",'-i',default="random",choices=["content","random"],
                    help="Way to initializa the generated image. Choose from 'content' and 'random'. 'content' initializes the image with content image. 'random' initializes the image with random noise. Default is 'random'.")
parser.add_argument("--print_iter",'-pi',type=int,default=1,
                    help="Print progress every 'print_iter' iterations. Set to 0 to disable printing. Default is 1.")
parser.add_argument("--save_iter",'-si',type=int,default=0,
                    help="Save intermediate images every 'save_iter' iterations. Set to 0 to disable saving intermediate images. Default is 0.")
args = parser.parse_args()
print(args)
import torch 
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable
import re,os
from PIL import Image

class GramMatrix(nn.Module):
    def forward(self, input):
        a,b,c,d = input.size() #N,C,H,W N=1
        feats = input.view(a*b,c*d) 
        gram = torch.mm(feats,feats.t())
        return gram.div(a*b*c*d)


class StyleNet(nn.Module):
    def __init__(self,args):
        super(StyleNet,self).__init__()
        self.style_img_names = args.sty_imgs
        self.content_img_name = args.con_img
        self.style_blend_weights = args.style_blend_weights
        if self.style_blend_weights!=None and len(self.style_blend_weights)!=len(self.style_img_names):
            raise ValueError("Length of style_blend_weights(%d) does not match the length of sty_imgs(%d)!"%(len(self.style_blend_weights),len(self.style_img_names)))
        self.out_path = args.out
        self.out_size = args.out_size 
        if self.out_size!=None and len(self.out_size)==1:
            self.out_size = self.out_size[0]
        self.content_layers = args.content_layers
        self.style_layers = args.style_layers
        self._layers_validation()
        self.MAX_C = self._get_max_layer(self.content_layers)
        self.MAX_S = self._get_max_layer(self.style_layers)
        self.content_weight = args.content_weight
        self.style_weight = args.style_weight
        self.optim = args.optim
        self.lr = args.learning_rate
        self.num_iter = args.num_iter
        self.pooling = args.pooling
        self.init = args.init
        self.print_iter = args.print_iter
        self.save_iter = args.save_iter
        if args.use_cuda:
            if torch.cuda.is_available():
                self.use_cuda = True
            else:
                raise ValueError("No valid CUDA device is found! You can remove the flag '--use_cuda' to run on CPU.")
        else:
            self.use_cuda = False
        self.dtype = torch.cuda.FloatTensor if self.use_cuda else torch.FloatTensor
        self.loss = nn.MSELoss().cuda() if self.use_cuda else nn.MSELoss()
        self.loss_net = self._load_vgg19()
        self.content_img = self._load_cont_img()
        self.style_imgs = self._load_sty_imgs()
        if self.init=="content":
            self.gen_img = nn.Parameter(self.content_img.data.clone().type(self.dtype))
        if self.init=='random':
            self.gen_img = nn.Parameter(torch.randn(self.content_img.size()).type(self.dtype))
        self.gram = self._load_gram()    
        if self.optim=='adam':
            self.optimizer = optim.Adam([self.gen_img],lr=self.lr)
        if self.optim=='lbfgs':
            self.optimizer = optim.LBFGS([self.gen_img],lr=self.lr)
            
    def _closure(self):
        def _replace(layer):
            tmp = layer
            if isinstance(layer,nn.ReLU):
                tmp = nn.ReLU(inplace=False)
            if isinstance(layer,nn.MaxPool2d) and self.pooling=="ave":
                tmp = nn.AvgPool2d(kernel_size=2,stride=2)
            if self.use_cuda:
                tmp = tmp.cuda()
            return tmp
        self.optimizer.zero_grad()
        self.gen_img.data.clamp_(0,1)
        gen_img = self.gen_img.clone()
        content_img = self.content_img.clone()
        style_imgs = [img.clone() for img in self.style_imgs]
       
        content_loss = 0
        style_loss = 0
        index = 1
        for layer in list(self.loss_net):
            layer = _replace(layer)
            if index<=self.MAX_C or index<=self.MAX_S:
                gen_img = layer(gen_img)
                if index<=self.MAX_C:
                    content_img = layer(content_img)
                if index<=self.MAX_S:
                    style_imgs = [layer(img) for img in style_imgs]
            else:
                break
            layer_name = "invalid_layer"
            if isinstance(layer,nn.Conv2d):
                layer_name = "conv_"+str(index)
            if isinstance(layer,nn.ReLU):
                layer_name = "relu_"+str(index)
                index += 1
            if layer_name in self.content_layers:
                content_loss += self.loss(gen_img*self.content_weight,content_img.detach()*self.content_weight)
            if layer_name in self.style_layers:
                gen_img_gram = self.gram(gen_img)
                style_imgs_grams = [self.gram(img) for img in style_imgs]
                style_losses = [self.loss(gen_img_gram*self.style_weight,sty_gram.detach()*self.style_weight)/len(self.style_layers) for sty_gram in style_imgs_grams]
                if self.style_blend_weights!=None:
                    assert len(style_losses)==len(self.style_blend_weights)
                    total = sum(self.style_blend_weights)
                    for i in xrange(len(style_losses)):
                        style_loss += self.style_blend_weights[i]*style_losses[i]/total
                else:
                    length = len(style_losses)
                    for i in xrange(length):
                        style_loss += style_losses[i]*1.0/length
        total_loss = content_loss + style_loss
        
        self.t_loss,self.c_loss,self.s_loss = total_loss,content_loss,style_loss
        total_loss.backward()
        if self.optim=="adam":
            self.optimizer.step()
        else:
            return total_loss
                      
    def train(self):
        if self.optim=="adam":
            self._closure()
        else:
            self.optimizer.step(self._closure)
            
    def run(self):
        for i in range(self.num_iter):

            self.train()
            if self.print_iter!=0 and (i+1)%self.print_iter==0:
                print("run {}:".format(i+1))
                print('Style Loss: {:4f} Content Loss: {:4f} Total_Loss: {:4f}'.format(
                    self.s_loss.data[0], self.c_loss.data[0], self.t_loss.data[0])) 
                print() 
            if self.save_iter!=0 and (i+1)%self.save_iter==0:
                out_folder = self.out_path.split(os.path.sep)[:-1]
                out_folder += ["temp_img_iter_%s.jpg"%(i+1)]
                filename = os.path.join(*out_folder)
                self._save_image(filename)
        self._save_image(self.out_path)
        
    def _save_image(self,filename):
        image = self.gen_img.data.cpu().clamp_(0,1)
        image = image.squeeze(0)
        image = transforms.ToPILImage()(image)
        image.save(filename)
        
    def _load_gram(self):
        gram = GramMatrix()
        if self.use_cuda:
            gram = gram.cuda()
        return gram
        
    def _load_cont_img(self):
        img = Image.open(self.content_img_name)
        if self.out_size!=None:
            img = transforms.Resize(self.out_size)(img)
        self._content_image_size = (img.size[1],img.size[0])## h,w of content image
        img = transforms.ToTensor()(img)
        img = Variable(img)
        img = img.unsqueeze(0)
        return img.type(self.dtype)
        
    def _load_sty_imgs(self):
        sty_imgs = []
        for sty_img_name in self.style_img_names:
            img = Image.open(sty_img_name)
            img = transforms.Resize(self._content_image_size)(img)
            img = transforms.ToTensor()(img)
            img = Variable(img)
            img = img.unsqueeze(0)
            assert img.size() ==  self.content_img.size()
            sty_imgs.append(img.type(self.dtype))
        return sty_imgs

    def _get_max_layer(self,layers):
        return max([int(l.split("_")[-1]) for l in layers])

    def _layers_validation(self):
        """
        Check the validity of the content layers and style layers specified by user. 
        """
        pattern = re.compile("^(conv|relu)_\d+$")
        for layer in self.content_layers+self.style_layers:
            if pattern.match(layer)==None or int(layer.split("_")[-1])>16 or int(layer.split("_")[-1])<1:
                raise ValueError("Invalid layer '%s' specified in vgg19. Please use format 'relu_i' or 'conv_i' where 1<=i<=16 for '--content_layers\-cls' and '--style_layers\-sls'."%layer)
    
    def _load_vgg19(self):
        model = models.vgg19(pretrained=True).features
        if self.use_cuda:
            model = model.cuda()
        return model
        
if __name__ == "__main__":
    style_net = StyleNet(args)
    style_net.run()
                