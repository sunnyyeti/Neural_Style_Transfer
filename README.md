# Neural_Style_Transfer
This is a pyTorch implementation of the paper [A Neural Algorithm of Artistic Style](https://arxiv.org/pdf/1508.06576.pdf) by Leon A. Gatys, Alexander S. Ecker, and Matthias Bethge, which proposes an algorithm to combine content of one image and style of another image. Here shows one example result I get using this implementation by combing a ballet dancer and one painting.

ballet dancer|painting|result
-------------|----------------|------
![](./content/dancing.jpg)|![](./style/1.jpg)|![](./output/dancing1.jpg)

With this algorithm, we can apply different styles from different images to the same content and get quite interesting results. Here I apply several styles to the Tubingen picture.

||<img src="./style/1.jpg" width="200" height="150">|<img src="./style/2.jpg" width="200" height="150">|<img src="./style/3.jpg" width="200" height="150">|<img src="./style/4.jpg" width="200" height="150">|
|---|---|---|---|---|
|<img src="./content/tubingen.jpg" width="200" height="150">|<img src="./output/tubingen_1.jpg" width="200" height="150">|<img src="./output/tubingen_2.jpg" width="200" height="150">|<img src="./output/tubingen_3.jpg" width="200" height="150">|<img src="./output/tubingen_4.jpg" width="200" height="150">|

We can also apply more than one style to the same content at the same time and the weights of different style can adjusted. Below is an example I apply two styles with weight ration 3:7 to the Tubingen image.

<table>
  <tr>
    <td></td>
    <td><img src="./style/1.jpg" width="200" height="150"></td>
    <td><img src="./style/3.jpg" width="200" height="150"></td>
  </tr>
  <tr>
    <td><img src="./content/tubingen.jpg" width="200" height="150"></td>
    <td colspan="2"><img src="./output/tubingen37.jpg" width="450" height="150"></td>
  </tr>
</table>

## Dependencies:
* Pytorch
* __Optional__:
    1. CUDA 
    2. cuDnn

## Usage:
Basic usage:
```
python NeuralStyle --sty_imgs <image.jpg> <image.jpg>... --con_img <image.jpg> --out <image.jpg>
```

