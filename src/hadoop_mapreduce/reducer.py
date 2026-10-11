#!/usr/bin/env python3
import sys

current_key = None
totals = [0] * 7

def output(key, values):
    if key is not None:
        print(key + "\t" + "\t".join(map(str, values)))

for line in sys.stdin:
    line = line.rstrip("\n")
    if "\t" not in line:
        continue

    key, value = line.split("\t", 1)
    try:
        numbers = list(map(int, value.split(",")))
    except ValueError:
        continue

    if len(numbers) != 7:
        continue

    if key != current_key:
        output(current_key, totals)
        current_key = key
        totals = [0] * 7

    totals = [a + b for a, b in zip(totals, numbers)]

output(current_key, totals)
