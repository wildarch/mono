# High-performance Arithmetic Expressions

## Task
You have a data stream of arithmetic expressions to STDIN.
Each line has one expression.
The expression contains 50.000 int16 numbers, basic operations: `+`, `-`, `*`, `/` (integer division) and `(`, `)`.
For example:

```
-19550 - ((-14208 / (13583 + -19215)) + (-16832 / 797 + (9060 / -23627)) + ((-6060) + 24953))
30835 - 3703 - (-20089 * -6261 + ((-28985 - 29627) + (-17828 - (22773 / -4014) * 1630)))
-14543 + (-12094 / -20726 + 25651 + (13732 - (28133))) * (1504) + -16348 + (-18371 - (-5750))
(14025) * 12700 + 14455 + ((25584) * -2310) + (27213 + (20470 + -14644 / 1949 / (-20039)))
(-21888 - 15779 - -2220) / 16967 + (20044 + -106 * ((10741 / -9574) * (-28909 - 3737)))
```

The number of rows is 100.

Evaluate each expression and print the results to STDOUT for the shortest time.
The result has int64 type.
