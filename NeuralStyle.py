# --style image: a list 
# --style_merge_weight 
# --content image 
# --output_size: default:same as input;int;tuple of int 
# --output_image:file path 
# --optim: Amda; LBFGS 
# --num_iter:int 
# --learning_rate:learning rate for Adam 
# --use_cuda:whether to use GPU 
# --content_layers 
# --style_layes 
# --content weight 
# --style weight 
# --pooling: max; ave 
# --init_img: from content; random 
# --print_iter: how often to print message 
# --save_iter:how often to save intermediate 

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
                    help="Weights of different style to blend. This should have the same length as option '--sty_img/-s'. Please separate weights by space. Equal weights are assigned to different if leave it default.")
parser.add_argument("--con_img","-c",required=True,
                    help="File path to the content image.")
parser.add_argument("--out","-o",required=True,
                    help="File path to the output image.")
parser.add_argument("--out_size",'-os',nargs="+",action=required_length(1,2),
                    help="Size of the output image. By default, the output image has the same size as content image. You can specify the height and width by seperating them with space like 'h w'. If you only specify an Int, then the size of the output is determined by matching the smaller edge of the content image to this number while keeping the ratio.")
parser.add_argument("--content_layers",'-cls',nargs="+",default=["conv_4"],
                    help="Layers used to reconstruct content. If there are more than one layer, specify them by space. Default is ['conv_4'].")
parser.add_argument("--style_layers",'-sls',nargs="+",default=['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5'],
                    help="Layers used to reconstruct style. If there are more than one layer, specify them by space. Default is ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5'].")
parser.add_argument("--content_weight","-cw",type=float,default=1.0,
                    help="Weight of the reconstruction of the content. Default is 1.0")
parser.add_argument("--style_weight","-sw",type=float,default=1e3,
                    help="Weight of the reconstruction of the style. Default is 1000.0.")
parser.add_argument("--optim",default="lbfgs",choices=['adam','lbfgs'],
                    help="Specify the optimization algorithm. Please choose from 'adam' and 'lbfgs'. Default is 'lbfgs'.")
parser.add_argument("--learning_rate",'-lr',default=10.0,type=float,
                    help="Learning rate for 'adam' optimization algorithm if '--optim' is specified as 'adam'. Otherwise, it is ignored. Default is 10.0.")
parser.add_argument("--num_iter",'-ni',default=300,type=int,
                    help="Number of the iterations. Default is 300.")
parser.add_argument("--use_cuda",action="store_true",
                    help="Switch to use CUDA to accelerate computing on GPU.")
parser.add_argument("--pooling","-p",default="max",choices=["max","ave"],
                    help="Type of pooling layer. Choose from 'max' and 'ave'. Default is 'max'.")
parser.add_argument("--init",'-i',default="content",choices=["content","random"],
                    help="Way to initializa the generated image. Choose from 'content' and 'random'. 'content' initializes the image with content image. 'random' initializes the image with random noise. Default is 'content'.")
parser.add_argument("--print_iter",type=int,default=0,
                    help="Print progress every 'print_iter' iterations. Set to 0 to disable printing. Default is 0.")
parser.add_argument("--save_iter",type=int,default=0,
                    help="Save intermediate images every 'save_iter' iterations. Set to 0 to disable saving intermediate images. Default is 0.")
args = parser.parse_args()

import torch 
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable
import warnings

class GramMatrix(nn.Module):
    def forward(self, input):
        a,b,c,d = input.size() #N,C,H,W
        feats = input.view(a*b,c*d) 
        gram = torch.mm(feats,feats.t())
        return gram.div(a*b*c*d)


class StyleNet(nn.Module):
    def __init__(self,args):
        self.style_imgs = args.sty_imgs
        self.content_img = args.con_img
        self.style_blend_weights = args.style_blend_weights
        if self.style_blend_weights!=None and len(self.style_blend_weights)!=len(self.sty_imgs):
            raise ValueError("Length of style_blend_weights(%d) does not match the length of sty_imgs(%d)!"%(len(self.style_blend_weights),len(self.sty_imgs)))
        self.out_path = args.out
        self.out_size = args.out_size
        self.content_layers = args.content_layers
        self.style_layers = args.style_layers
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