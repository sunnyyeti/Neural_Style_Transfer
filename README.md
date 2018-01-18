# Neural_Style_Transfer
This is a pyTorch implementation of the paper [A Neural Algorithm of Artistic Style](https://arxiv.org/pdf/1508.06576.pdf) by Leon A. Gatys, Alexander S. Ecker, and Matthias Bethge, which proposes an algorithm to combine content of one image and style of another image. Here shows one example result I get using this implementation by combing a ballet dancer and one painting.

ballet dancer|painting|result
-------------|----------------|------
![](./content/dancing.jpg)|![](./style/1.jpg)|![](./output/dancing1.jpg)

With this algorithm, we can apply different styles from different images to the same content and get quite interesting results. Here I apply several styles to the Tubingen picture.

||<img src="./style/1.jpg" width="200" height="150">|<img src="./style/2.jpg" width="200" height="150">|<img src="./style/3.jpg" width="200" height="150">|<img src="./style/4.jpg" width="200" height="150">|
|---|---|---|---|---|
|<img src="./content/tubingen.jpg" width="150" height="80">|<img src="./output/tubingen_1.jpg" width="150" height="80">|<img src="./output/tubingen_2.jpg" width="150" height="80">|<img src="./output/tubingen_3.jpg" width="150" height="80">|<img src="./output/tubingen_4.jpg" width="150" height="80">|

fdf

