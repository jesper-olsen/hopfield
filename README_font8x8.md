## Font8x8
=============

Train network on letters a,b,c (8x8 font = 64-bit images); recognise from initialisation a,b,c,d,e,f - repeat 3 times:
* no mask
* mask upper 32 bits.
* mask lower 32 bits.

Run the example:

```
% cargo run --bin main_font8x8


```
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

Goodness: 420
Goodness: 420
Goodness: 420
Goodness: 420
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

Goodness: 560
Goodness: 560
Goodness: 560
Goodness: 560
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

Goodness: 386
Goodness: 386
Goodness: 386
Goodness: 386
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

Goodness: -48
Goodness: 420
Goodness: 420
Goodness: 420
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

Goodness: 244
Goodness: 418
Goodness: 418
Goodness: 418
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

Goodness: 52
Goodness: 1964
Goodness: 3596
Goodness: 3596
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

Goodness: 58
Goodness: 418
Goodness: 418
Goodness: 418
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

Goodness: 128
Goodness: 560
Goodness: 560
Goodness: 560
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

Goodness: 60
Goodness: 386
Goodness: 386
Goodness: 386
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

Goodness: 20
Goodness: 3596
Goodness: 3596
Goodness: 3596
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

Goodness: 60
Goodness: 386
Goodness: 386
Goodness: 386
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

Goodness: 76
Goodness: 3424
Goodness: 3596
Goodness: 3596
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

Goodness: 162
Goodness: 420
Goodness: 420
Goodness: 420
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

Goodness: 140
Goodness: 560
Goodness: 560
Goodness: 560
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

Goodness: 110
Goodness: 402
Goodness: 418
Goodness: 418
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

Goodness: 108
Goodness: 420
Goodness: 420
Goodness: 420
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

Goodness: 48
Goodness: 420
Goodness: 420
Goodness: 420
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

Goodness: 32
Goodness: 418
Goodness: 418
Goodness: 418
🟨🟨🟨🟨🟨🟨🟨🟨
🟨🟨🟨🟨🟨🟨🟨🟨
🟨🟦🟦🟦🟦🟨🟨🟨
🟨🟦🟨🟨🟦🟦🟨🟨
🟨🟦🟦🟨🟨🟦🟨🟨
🟦🟦🟨🟨🟦🟦🟨🟨
🟨🟦🟦🟦🟦🟦🟨🟨
🟨🟨🟨🟨🟨🟨🟨🟨
```
