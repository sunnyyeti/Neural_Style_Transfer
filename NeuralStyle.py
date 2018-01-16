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

parser = argparse.ArgumentParser(description="Generate an image with content as the 'content' image and style as the 'style' image(s) you provide.")
parser.add_argument("--sty_img","-s",nargs="+",required=True, help="File path(es) to the style image(s). If there are more than one style image, please separate them by space.")
parser.add_argument("--con_img","-c",nargs=1,required=True,help="File path to the content image")
args = parser.parse_args()
#parser.parse_args("-s a b c -c ".split())
print args

 