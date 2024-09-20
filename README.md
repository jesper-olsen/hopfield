Hopfield Networks
=================

Train classic Hopfield networks [1,2,3] to store memories (bit vectors) using Hopfield's one-shot storage rule or the iterative perceptron convergence procedure.

Two examples below 
* [Font8x8](#Font8x8) - small number of static memories (8x8 bitmaps).
* [MNIST](#MNIST) - MNIST handwritten digit classification.

References:
-----------
[1] [Neural networks and physical systems with emergent collective computational abilities, J.J. Hopfield, Proc. Natl. Acad. Sci., Vol 79, pp2554-2558, April 1982](https://www.pnas.org/doi/epdf/10.1073/pnas.79.8.2554) <br/>
[2] [Hinton's Hopfield video 11a](https://www.cs.toronto.edu/~hinton/coursera/lecture11/lec11a.mp4) <br/>
[3] [Hinton's Hopfield video 11b](https://www.cs.toronto.edu/~hinton/coursera/lecture11/lec11b.mp4) <br/>


## Font8x8
------------

Run:
----
Train network on letters a,b,c (8x8 font = 64-bit images); recognise from initialisation a,b,c,d,e,f - repeat 3 times:
* no mask
* mask upper 32 bits.
* mask lower 32 bits.
```
% cargo run --bin main_font8x8

🟨🟨🟨🟨🟨🟨🟨🟨
🟨🟨🟨🟨🟨🟨🟨🟨
🟨🟦🟦🟦🟦🟨🟨🟨
🟨🟨🟨🟨🟦🟦🟨🟨
🟨🟦🟦🟦🟦🟦🟨🟨
🟦🟦🟨🟨🟦🟦🟨🟨
🟨🟦🟦🟦🟨🟦🟦🟨
🟨🟨🟨🟨🟨🟨🟨🟨

🟦🟦🟦🟨🟨🟨🟨🟨
🟨🟦🟦🟨🟨🟨🟨🟨
🟨🟦🟦🟨🟨🟨🟨🟨
🟨🟦🟦🟦🟦🟦🟨🟨
🟨🟦🟦🟨🟨🟦🟦🟨
🟨🟦🟦🟨🟨🟦🟦🟨
🟦🟦🟨🟦🟦🟦🟨🟨
🟨🟨🟨🟨🟨🟨🟨🟨

🟨🟨🟨🟨🟨🟨🟨🟨
🟨🟨🟨🟨🟨🟨🟨🟨
🟨🟦🟦🟦🟦🟨🟨🟨
🟦🟦🟨🟨🟦🟦🟨🟨
🟦🟦🟨🟨🟨🟨🟨🟨
🟦🟦🟨🟨🟦🟦🟨🟨
🟨🟦🟦🟦🟦🟨🟨🟨
🟨🟨🟨🟨🟨🟨🟨🟨

Initialising with a - mask: none
🟨🟨🟨🟨🟨🟨🟨🟨
🟨🟨🟨🟨🟨🟨🟨🟨
🟨🟦🟦🟦🟦🟨🟨🟨
🟨🟨🟨🟨🟦🟦🟨🟨
🟨🟦🟦🟦🟦🟦🟨🟨
🟦🟦🟨🟨🟦🟦🟨🟨
🟨🟦🟦🟦🟨🟦🟦🟨
🟨🟨🟨🟨🟨🟨🟨🟨

Goodness: 452
Goodness: 452
Goodness: 452
Goodness: 452
🟨🟨🟨🟨🟨🟨🟨🟨
🟨🟨🟨🟨🟨🟨🟨🟨
🟨🟦🟦🟦🟦🟨🟨🟨
🟨🟨🟨🟨🟦🟦🟨🟨
🟨🟦🟦🟦🟦🟦🟨🟨
🟦🟦🟨🟨🟦🟦🟨🟨
🟨🟦🟦🟦🟨🟦🟦🟨
🟨🟨🟨🟨🟨🟨🟨🟨

Initialising with b - mask: none
🟦🟦🟦🟨🟨🟨🟨🟨
🟨🟦🟦🟨🟨🟨🟨🟨
🟨🟦🟦🟨🟨🟨🟨🟨
🟨🟦🟦🟦🟦🟦🟨🟨
🟨🟦🟦🟨🟨🟦🟦🟨
🟨🟦🟦🟨🟨🟦🟦🟨
🟦🟦🟨🟦🟦🟦🟨🟨
🟨🟨🟨🟨🟨🟨🟨🟨

Goodness: 581
Goodness: 581
Goodness: 581
Goodness: 581
🟦🟦🟦🟨🟨🟨🟨🟨
🟨🟦🟦🟨🟨🟨🟨🟨
🟨🟦🟦🟨🟨🟨🟨🟨
🟨🟦🟦🟦🟦🟦🟨🟨
🟨🟦🟦🟨🟨🟦🟦🟨
🟨🟦🟦🟨🟨🟦🟦🟨
🟦🟦🟨🟦🟦🟦🟨🟨
🟨🟨🟨🟨🟨🟨🟨🟨

Initialising with c - mask: none
🟨🟨🟨🟨🟨🟨🟨🟨
🟨🟨🟨🟨🟨🟨🟨🟨
🟨🟦🟦🟦🟦🟨🟨🟨
🟦🟦🟨🟨🟦🟦🟨🟨
🟦🟦🟨🟨🟨🟨🟨🟨
🟦🟦🟨🟨🟦🟦🟨🟨
🟨🟦🟦🟦🟦🟨🟨🟨
🟨🟨🟨🟨🟨🟨🟨🟨

Goodness: 418
Goodness: 418
Goodness: 418
Goodness: 418
🟨🟨🟨🟨🟨🟨🟨🟨
🟨🟨🟨🟨🟨🟨🟨🟨
🟨🟦🟦🟦🟦🟨🟨🟨
🟦🟦🟨🟨🟦🟦🟨🟨
🟦🟦🟨🟨🟨🟨🟨🟨
🟦🟦🟨🟨🟦🟦🟨🟨
🟨🟦🟦🟦🟦🟨🟨🟨
🟨🟨🟨🟨🟨🟨🟨🟨

Initialising with d - mask: none
🟨🟨🟨🟦🟦🟦🟨🟨
🟨🟨🟨🟨🟦🟦🟨🟨
🟨🟨🟨🟨🟦🟦🟨🟨
🟨🟦🟦🟦🟦🟦🟨🟨
🟦🟦🟨🟨🟦🟦🟨🟨
🟦🟦🟨🟨🟦🟦🟨🟨
🟨🟦🟦🟦🟨🟦🟦🟨
🟨🟨🟨🟨🟨🟨🟨🟨

Goodness: -43
Goodness: 452
Goodness: 452
Goodness: 452
🟨🟨🟨🟨🟨🟨🟨🟨
🟨🟨🟨🟨🟨🟨🟨🟨
🟨🟦🟦🟦🟦🟨🟨🟨
🟨🟨🟨🟨🟦🟦🟨🟨
🟨🟦🟦🟦🟦🟦🟨🟨
🟦🟦🟨🟨🟦🟦🟨🟨
🟨🟦🟦🟦🟨🟦🟦🟨
🟨🟨🟨🟨🟨🟨🟨🟨

Initialising with e - mask: none
🟨🟨🟨🟨🟨🟨🟨🟨
🟨🟨🟨🟨🟨🟨🟨🟨
🟨🟦🟦🟦🟦🟨🟨🟨
🟦🟦🟨🟨🟦🟦🟨🟨
🟦🟦🟦🟦🟦🟦🟨🟨
🟦🟦🟨🟨🟨🟨🟨🟨
🟨🟦🟦🟦🟦🟨🟨🟨
🟨🟨🟨🟨🟨🟨🟨🟨

Goodness: 272
Goodness: 455
Goodness: 455
Goodness: 455
🟨🟨🟨🟨🟨🟨🟨🟨
🟨🟨🟨🟨🟨🟨🟨🟨
🟨🟦🟦🟦🟦🟨🟨🟨
🟨🟦🟨🟨🟦🟦🟨🟨
🟨🟦🟦🟨🟨🟦🟨🟨
🟦🟦🟨🟨🟦🟦🟨🟨
🟨🟦🟦🟦🟦🟦🟨🟨
🟨🟨🟨🟨🟨🟨🟨🟨

Initialising with f - mask: none
🟨🟨🟦🟦🟦🟨🟨🟨
🟨🟦🟦🟨🟦🟦🟨🟨
🟨🟦🟦🟨🟨🟨🟨🟨
🟦🟦🟦🟦🟨🟨🟨🟨
🟨🟦🟦🟨🟨🟨🟨🟨
🟨🟦🟦🟨🟨🟨🟨🟨
🟦🟦🟦🟦🟨🟨🟨🟨
🟨🟨🟨🟨🟨🟨🟨🟨

Goodness: 53
Goodness: 1895
Goodness: 3493
Goodness: 3493
🟦🟦🟦🟦🟦🟦🟦🟦
🟦🟦🟦🟦🟦🟦🟦🟦
🟦🟨🟨🟨🟨🟦🟦🟦
🟦🟨🟦🟦🟨🟨🟦🟦
🟦🟨🟨🟦🟦🟨🟦🟦
🟨🟨🟦🟦🟨🟨🟦🟦
🟦🟨🟨🟨🟨🟨🟦🟦
🟦🟦🟦🟦🟦🟦🟦🟦

Initialising with a - mask: upper
🟨🟨🟨🟨🟨🟨🟨🟨
🟨🟨🟨🟨🟨🟨🟨🟨
🟨🟦🟦🟦🟦🟨🟨🟨
🟨🟨🟨🟨🟦🟦🟨🟨
🟨🟨🟨🟨🟨🟨🟨🟨
🟨🟨🟨🟨🟨🟨🟨🟨
🟨🟨🟨🟨🟨🟨🟨🟨
🟨🟨🟨🟨🟨🟨🟨🟨

Goodness: 72
Goodness: 455
Goodness: 455
Goodness: 455
🟨🟨🟨🟨🟨🟨🟨🟨
🟨🟨🟨🟨🟨🟨🟨🟨
🟨🟦🟦🟦🟦🟨🟨🟨
🟨🟦🟨🟨🟦🟦🟨🟨
🟨🟦🟦🟨🟨🟦🟨🟨
🟦🟦🟨🟨🟦🟦🟨🟨
🟨🟦🟦🟦🟦🟦🟨🟨
🟨🟨🟨🟨🟨🟨🟨🟨

Initialising with b - mask: upper
🟦🟦🟦🟨🟨🟨🟨🟨
🟨🟦🟦🟨🟨🟨🟨🟨
🟨🟦🟦🟨🟨🟨🟨🟨
🟨🟦🟦🟦🟦🟦🟨🟨
🟨🟨🟨🟨🟨🟨🟨🟨
🟨🟨🟨🟨🟨🟨🟨🟨
🟨🟨🟨🟨🟨🟨🟨🟨
🟨🟨🟨🟨🟨🟨🟨🟨

Goodness: 134
Goodness: 581
Goodness: 581
Goodness: 581
🟦🟦🟦🟨🟨🟨🟨🟨
🟨🟦🟦🟨🟨🟨🟨🟨
🟨🟦🟦🟨🟨🟨🟨🟨
🟨🟦🟦🟦🟦🟦🟨🟨
🟨🟦🟦🟨🟨🟦🟦🟨
🟨🟦🟦🟨🟨🟦🟦🟨
🟦🟦🟨🟦🟦🟦🟨🟨
🟨🟨🟨🟨🟨🟨🟨🟨

Initialising with c - mask: upper
🟨🟨🟨🟨🟨🟨🟨🟨
🟨🟨🟨🟨🟨🟨🟨🟨
🟨🟦🟦🟦🟦🟨🟨🟨
🟦🟦🟨🟨🟦🟦🟨🟨
🟨🟨🟨🟨🟨🟨🟨🟨
🟨🟨🟨🟨🟨🟨🟨🟨
🟨🟨🟨🟨🟨🟨🟨🟨
🟨🟨🟨🟨🟨🟨🟨🟨

Goodness: 74
Goodness: 418
Goodness: 418
Goodness: 418
🟨🟨🟨🟨🟨🟨🟨🟨
🟨🟨🟨🟨🟨🟨🟨🟨
🟨🟦🟦🟦🟦🟨🟨🟨
🟦🟦🟨🟨🟦🟦🟨🟨
🟦🟦🟨🟨🟨🟨🟨🟨
🟦🟦🟨🟨🟦🟦🟨🟨
🟨🟦🟦🟦🟦🟨🟨🟨
🟨🟨🟨🟨🟨🟨🟨🟨

Initialising with d - mask: upper
🟨🟨🟨🟦🟦🟦🟨🟨
🟨🟨🟨🟨🟦🟦🟨🟨
🟨🟨🟨🟨🟦🟦🟨🟨
🟨🟦🟦🟦🟦🟦🟨🟨
🟨🟨🟨🟨🟨🟨🟨🟨
🟨🟨🟨🟨🟨🟨🟨🟨
🟨🟨🟨🟨🟨🟨🟨🟨
🟨🟨🟨🟨🟨🟨🟨🟨

Goodness: 8
Goodness: 3493
Goodness: 3493
Goodness: 3493
🟦🟦🟦🟦🟦🟦🟦🟦
🟦🟦🟦🟦🟦🟦🟦🟦
🟦🟨🟨🟨🟨🟦🟦🟦
🟦🟨🟦🟦🟨🟨🟦🟦
🟦🟨🟨🟦🟦🟨🟦🟦
🟨🟨🟦🟦🟨🟨🟦🟦
🟦🟨🟨🟨🟨🟨🟦🟦
🟦🟦🟦🟦🟦🟦🟦🟦

Initialising with e - mask: upper
🟨🟨🟨🟨🟨🟨🟨🟨
🟨🟨🟨🟨🟨🟨🟨🟨
🟨🟦🟦🟦🟦🟨🟨🟨
🟦🟦🟨🟨🟦🟦🟨🟨
🟨🟨🟨🟨🟨🟨🟨🟨
🟨🟨🟨🟨🟨🟨🟨🟨
🟨🟨🟨🟨🟨🟨🟨🟨
🟨🟨🟨🟨🟨🟨🟨🟨

Goodness: 74
Goodness: 418
Goodness: 418
Goodness: 418
🟨🟨🟨🟨🟨🟨🟨🟨
🟨🟨🟨🟨🟨🟨🟨🟨
🟨🟦🟦🟦🟦🟨🟨🟨
🟦🟦🟨🟨🟦🟦🟨🟨
🟦🟦🟨🟨🟨🟨🟨🟨
🟦🟦🟨🟨🟦🟦🟨🟨
🟨🟦🟦🟦🟦🟨🟨🟨
🟨🟨🟨🟨🟨🟨🟨🟨

Initialising with f - mask: upper
🟨🟨🟦🟦🟦🟨🟨🟨
🟨🟦🟦🟨🟦🟦🟨🟨
🟨🟦🟦🟨🟨🟨🟨🟨
🟦🟦🟦🟦🟨🟨🟨🟨
🟨🟨🟨🟨🟨🟨🟨🟨
🟨🟨🟨🟨🟨🟨🟨🟨
🟨🟨🟨🟨🟨🟨🟨🟨
🟨🟨🟨🟨🟨🟨🟨🟨

Goodness: 65
Goodness: 3326
Goodness: 3493
Goodness: 3493
🟦🟦🟦🟦🟦🟦🟦🟦
🟦🟦🟦🟦🟦🟦🟦🟦
🟦🟨🟨🟨🟨🟦🟦🟦
🟦🟨🟦🟦🟨🟨🟦🟦
🟦🟨🟨🟦🟦🟨🟦🟦
🟨🟨🟦🟦🟨🟨🟦🟦
🟦🟨🟨🟨🟨🟨🟦🟦
🟦🟦🟦🟦🟦🟦🟦🟦

Initialising with a - mask: lower
🟨🟨🟨🟨🟨🟨🟨🟨
🟨🟨🟨🟨🟨🟨🟨🟨
🟨🟨🟨🟨🟨🟨🟨🟨
🟨🟨🟨🟨🟨🟨🟨🟨
🟨🟦🟦🟦🟦🟦🟨🟨
🟦🟦🟨🟨🟦🟦🟨🟨
🟨🟦🟦🟦🟨🟦🟦🟨
🟨🟨🟨🟨🟨🟨🟨🟨

Goodness: 180
Goodness: 452
Goodness: 452
Goodness: 452
🟨🟨🟨🟨🟨🟨🟨🟨
🟨🟨🟨🟨🟨🟨🟨🟨
🟨🟦🟦🟦🟦🟨🟨🟨
🟨🟨🟨🟨🟦🟦🟨🟨
🟨🟦🟦🟦🟦🟦🟨🟨
🟦🟦🟨🟨🟦🟦🟨🟨
🟨🟦🟦🟦🟨🟦🟦🟨
🟨🟨🟨🟨🟨🟨🟨🟨

Initialising with b - mask: lower
🟨🟨🟨🟨🟨🟨🟨🟨
🟨🟨🟨🟨🟨🟨🟨🟨
🟨🟨🟨🟨🟨🟨🟨🟨
🟨🟨🟨🟨🟨🟨🟨🟨
🟨🟦🟦🟨🟨🟦🟦🟨
🟨🟦🟦🟨🟨🟦🟦🟨
🟦🟦🟨🟦🟦🟦🟨🟨
🟨🟨🟨🟨🟨🟨🟨🟨

Goodness: 155
Goodness: 581
Goodness: 581
Goodness: 581
🟦🟦🟦🟨🟨🟨🟨🟨
🟨🟦🟦🟨🟨🟨🟨🟨
🟨🟦🟦🟨🟨🟨🟨🟨
🟨🟦🟦🟦🟦🟦🟨🟨
🟨🟦🟦🟨🟨🟦🟦🟨
🟨🟦🟦🟨🟨🟦🟦🟨
🟦🟦🟨🟦🟦🟦🟨🟨
🟨🟨🟨🟨🟨🟨🟨🟨

Initialising with c - mask: lower
🟨🟨🟨🟨🟨🟨🟨🟨
🟨🟨🟨🟨🟨🟨🟨🟨
🟨🟨🟨🟨🟨🟨🟨🟨
🟨🟨🟨🟨🟨🟨🟨🟨
🟦🟦🟨🟨🟨🟨🟨🟨
🟦🟦🟨🟨🟦🟦🟨🟨
🟨🟦🟦🟦🟦🟨🟨🟨
🟨🟨🟨🟨🟨🟨🟨🟨

Goodness: 128
Goodness: 438
Goodness: 455
Goodness: 455
🟨🟨🟨🟨🟨🟨🟨🟨
🟨🟨🟨🟨🟨🟨🟨🟨
🟨🟦🟦🟦🟦🟨🟨🟨
🟨🟦🟨🟨🟦🟦🟨🟨
🟨🟦🟦🟨🟨🟦🟨🟨
🟦🟦🟨🟨🟦🟦🟨🟨
🟨🟦🟦🟦🟦🟦🟨🟨
🟨🟨🟨🟨🟨🟨🟨🟨

Initialising with d - mask: lower
🟨🟨🟨🟨🟨🟨🟨🟨
🟨🟨🟨🟨🟨🟨🟨🟨
🟨🟨🟨🟨🟨🟨🟨🟨
🟨🟨🟨🟨🟨🟨🟨🟨
🟦🟦🟨🟨🟦🟦🟨🟨
🟦🟦🟨🟨🟦🟦🟨🟨
🟨🟦🟦🟦🟨🟦🟦🟨
🟨🟨🟨🟨🟨🟨🟨🟨

Goodness: 125
Goodness: 452
Goodness: 452
Goodness: 452
🟨🟨🟨🟨🟨🟨🟨🟨
🟨🟨🟨🟨🟨🟨🟨🟨
🟨🟦🟦🟦🟦🟨🟨🟨
🟨🟨🟨🟨🟦🟦🟨🟨
🟨🟦🟦🟦🟦🟦🟨🟨
🟦🟦🟨🟨🟦🟦🟨🟨
🟨🟦🟦🟦🟨🟦🟦🟨
🟨🟨🟨🟨🟨🟨🟨🟨

Initialising with e - mask: lower
🟨🟨🟨🟨🟨🟨🟨🟨
🟨🟨🟨🟨🟨🟨🟨🟨
🟨🟨🟨🟨🟨🟨🟨🟨
🟨🟨🟨🟨🟨🟨🟨🟨
🟦🟦🟦🟦🟦🟦🟨🟨
🟦🟦🟨🟨🟨🟨🟨🟨
🟨🟦🟦🟦🟦🟨🟨🟨
🟨🟨🟨🟨🟨🟨🟨🟨

Goodness: 62
Goodness: 452
Goodness: 452
Goodness: 452
🟨🟨🟨🟨🟨🟨🟨🟨
🟨🟨🟨🟨🟨🟨🟨🟨
🟨🟦🟦🟦🟦🟨🟨🟨
🟨🟨🟨🟨🟦🟦🟨🟨
🟨🟦🟦🟦🟦🟦🟨🟨
🟦🟦🟨🟨🟦🟦🟨🟨
🟨🟦🟦🟦🟨🟦🟦🟨
🟨🟨🟨🟨🟨🟨🟨🟨

Initialising with f - mask: lower
🟨🟨🟨🟨🟨🟨🟨🟨
🟨🟨🟨🟨🟨🟨🟨🟨
🟨🟨🟨🟨🟨🟨🟨🟨
🟨🟨🟨🟨🟨🟨🟨🟨
🟨🟦🟦🟨🟨🟨🟨🟨
🟨🟦🟦🟨🟨🟨🟨🟨
🟦🟦🟦🟦🟨🟨🟨🟨
🟨🟨🟨🟨🟨🟨🟨🟨

Goodness: 44
Goodness: 455
Goodness: 455
Goodness: 455
🟨🟨🟨🟨🟨🟨🟨🟨
🟨🟨🟨🟨🟨🟨🟨🟨
🟨🟦🟦🟦🟦🟨🟨🟨
🟨🟦🟨🟨🟦🟦🟨🟨
🟨🟦🟦🟨🟨🟦🟨🟨
🟦🟦🟨🟨🟦🟦🟨🟨
🟨🟦🟦🟦🟦🟦🟨🟨
🟨🟨🟨🟨🟨🟨🟨🟨

```

## MNIST
------------

```
% cargo run --bin main_mnist
```

The MNIST images are 8-bit monochrome - the Hopfield state is binary and intensity is here represented with a one-hot encodeing using 1, 2 or 3 bits.
The quantization level turns out to have little influence on accuracy:

| Quantisation | Correcty |
| 3 bits / 8 levels | 7833 |
| 2 bits / 4 levels | 8045 |
| 1 bit  / 2 levels | 8003 |

Accuracy is ~80% - the [Forward-Forward algorithm](https://github.com/jesper-olsen/ff-py) and  
[Deep Boltzmann machines](https://github.com/jesper-olsen/rbm-py) achieve better than 98% on the same task - though at the cost
of considerably more complexity.

Here is an example image from the test set:
![PNG](https://raw.githubusercontent.com/jesper-olsen/hopfield/master/7-org.png) 

And same image after being extracted from the Hopfield network - 1 bit quantisation:

![PNG](https://raw.githubusercontent.com/jesper-olsen/hopfield/master/7-q2.png) 

and 2-bit quantisation:

![PNG](https://raw.githubusercontent.com/jesper-olsen/hopfield/master/7-q4.png) 

and 3-bit quantisation:

![PNG](https://raw.githubusercontent.com/jesper-olsen/hopfield/master/7-q8.png) 



