# Homework 1

- Student ID: 309553002
- Name: 林育愷

## Prerequisites

Python 3.6^ involving following packages:

- `numpy`
- `matplotlib`

Note that f-string is used so version <= 3.5 is not allowed.

## Usage

```txt
$ python3 HW01.py --help
usage: HW01.py [-h] filename n regularizer

positional arguments:
  filename     file which consists of data points
  n            degree
  regularizer  regularizer (only for LSE)

optional arguments:
  -h, --help   show this help message and exit
```

For example,

```txt
$ python3 HW01.py input.txt 3 0
LSE:
Fitting line: 3.02385X^2 + 4.90619X^1 + -0.23140
Total error: 26.55996

Newton's Method:
Fitting line: 3.02385X^2 + 4.90619X^1 + -0.23140
Total error: 26.55996
```
