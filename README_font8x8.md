## Font8x8
=============

Train network on letters a,b,c (8x8 font = 64-bit images); recognise from initialisation a,b,c,d,e,f - repeat 3 times:
* no mask
* mask upper 32 bits.
* mask lower 32 bits.

Run the example:

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
