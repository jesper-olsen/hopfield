Hopfield Networks
=================

Train classic Hopfield networks [1,2,3,4] to store memories (bit vectors) using Hopfield's one-shot storage rule or the iterative perceptron convergence procedure.

References:
-----------
[1] [Neural networks and physical systems with emergent collective computational abilities, J.J. Hopfield, Proc. Natl. Acad. Sci., Vol 79, pp2554-2558, April 1982](https://www.pnas.org/doi/epdf/10.1073/pnas.79.8.2554) <br/>
[2] [Hinton's Hopfield video 11a](https://www.cs.toronto.edu/~hinton/coursera/lecture11/lec11a.mp4) <br/>
[3] [Hinton's Hopfield video 11b](https://www.cs.toronto.edu/~hinton/coursera/lecture11/lec11b.mp4) <br/>

Run:
----
Train network on letters a,b,c (8x8 font = 64-bit images); recognise from initialisation a,b,c,d,e,f.
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

Initialising with a
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

Initialising with b
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

Initialising with c
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

Initialising with d
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

Initialising with e
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

Initialising with f
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

```
