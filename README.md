Hopfield Networks
=================

Train a classic Hopfield networks [1,2,3,4] to store memories (bit vectors) using Hopfield's one-shot storage rule:

    Hopfield with -1 & 1 states
        delta w_ij = s_i * s_j
    Hopfield with 0 & 1 states
        delta w_ij = 4(s_i-0.5)(s_j-0.5)
                   = 4(s_i*s_j -0.5s_i-0.5*s_j+0.25)
                   = 4s_i*s_j -2s_i -2s_j + 1
    For M memories, weights in range [-M;M] 

or with the iterative  perceptron convergence procedure:
    * if output unit is correct do nothing
    * if incorrectly outputs zero, add input vector to weight vector
    * if incorrectly outputs one, subtract input vector from weight vector 

References:
-----------
[1] [Neural networks and physical systems with emergent collective computational abilities, J.J. Hopfield, Proc. Natl. Acad. Sci., Vol 79, pp2554-2558, April 1982](https://www.pnas.org/doi/epdf/10.1073/pnas.79.8.2554) <br/>
[2] [Hinton's coursera lecture 11a](https://www.cs.toronto.edu/~hinton/coursera/lecture11/lec11a.mp4) <br/>
[3] [Hinton's coursera lecture 11b](https://www.cs.toronto.edu/~hinton/coursera/lecture11/lec11b.mp4) <br/>
[4] [CMU Lectures 20-22](https://youtu.be/3Cp_pjPRmt8?si=VXghIuz-V9rDdQGN) <br/>

Run:
----
Train network on letters a,b,c (8x8 font = 64-bit images); recognise from initialisation d,e,f.
```
% cargo run

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

Initialising with d
Goodness0: -43
Goodness1: 452
Goodness2: 452
🟨🟨🟨🟨🟨🟨🟨🟨
🟨🟨🟨🟨🟨🟨🟨🟨
🟨🟦🟦🟦🟦🟨🟨🟨
🟨🟨🟨🟨🟦🟦🟨🟨
🟨🟦🟦🟦🟦🟦🟨🟨
🟦🟦🟨🟨🟦🟦🟨🟨
🟨🟦🟦🟦🟨🟦🟦🟨
🟨🟨🟨🟨🟨🟨🟨🟨

Initialising with e
Goodness0: 272
Goodness1: 455
Goodness2: 455
🟨🟨🟨🟨🟨🟨🟨🟨
🟨🟨🟨🟨🟨🟨🟨🟨
🟨🟦🟦🟦🟦🟨🟨🟨
🟨🟦🟨🟨🟦🟦🟨🟨
🟨🟦🟦🟨🟨🟦🟨🟨
🟦🟦🟨🟨🟦🟦🟨🟨
🟨🟦🟦🟦🟦🟦🟨🟨
🟨🟨🟨🟨🟨🟨🟨🟨

Initialising with f
Goodness0: 53
Goodness1: 1895
Goodness2: 3493
🟦🟦🟦🟦🟦🟦🟦🟦
🟦🟦🟦🟦🟦🟦🟦🟦
🟦🟨🟨🟨🟨🟦🟦🟦
🟦🟨🟦🟦🟨🟨🟦🟦
🟦🟨🟨🟦🟦🟨🟦🟦
🟨🟨🟦🟦🟨🟨🟦🟦
🟦🟨🟨🟨🟨🟨🟦🟦
🟦🟦🟦🟦🟦🟦🟦🟦
```
