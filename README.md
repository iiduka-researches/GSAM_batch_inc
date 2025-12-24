# CONVERGENCE OF SHARPNESSAWARE MINIMIZATION ALGORITHMS USING INCREASING BATCH SIZE AND DECAYING LEARNING RATE


### ABSTRACT
 
The sharpness-aware minimization (SAM) algorithm and its variants, including gap guided SAM (GSAM), have been successful at improving the generalization capability of deep neural network models by finding flat local minima of the empirical loss in training. Meanwhile, it has been shown theoretically and practically that increasing the batch size or decaying the learning rate avoids sharp local minima of the empirical loss. In this paper, we consider the GSAM algorithm with increasing batch sizes or decaying learning rates, such as cosine annealing or linear learning rate, and theoretically show its convergence. Moreover, we numerically compare SAM (GSAM) with and without an increasing batch size and conclude that using an increasing batch size or decaying learning rate finds flatter local minima than using a constant batch size and learning rate.

### Algorithm Performance(Cifar10)

**ResNet-18**
|               |    SGD |   SAM |   GSAM |   SGD + B |   SAM + B |   GSAM + B |   SGD + C |   SAM + C |   GSAM + C |
|:--------------|-------:|------:|-------:|----------:|----------:|-----------:|----------:|----------:|-----------:|
| Test Error(%) |   7.17 |  6.34 |   6.21 |      6.39 |      6.03 |       6.06 |      6.63 |      6.08 |       6.20 |
| Sharpness     | 106.46 | 51.82 |  48.39 |      8.73 |      4.27 |       4.08 |    141.37 |     88.53 |      89.30 |

**Wide-ResNet28-10**
|               |     SGD |    SAM |   GSAM |   SGD + B |   SAM + B |   GSAM + B |   SGD + C |   SAM + C |   GSAM + C |
|:--------------|--------:|-------:|-------:|----------:|----------:|-----------:|----------:|----------:|-----------:|
| Test Error(%) |    7.04 |   6.13 |   5.98 |      5.53 |      5.10 |       5.05 |      6.67 |      5.87 |       5.92 |
| Sharpness     | 1124.45 | 546.93 | 543.27 |     48.25 |     38.79 |      40.28 |   1467.67 |    771.79 |     794.45 |

### Algorithm Performance(Cifar100)

**ResNet-18**
|               | SGD    | SAM   | GSAM  | SGD + B | SAM + B | GSAM + B | SGD + C | SAM + C | GSAM + C | 
| :-----------: | :----: | :---: | :---: | :-----: | :-----: | :------: | :-----: | :-----: | :------: | 
| Test Error(%) | 26.61  | 26.39 | 26.61 | 25.58   | 25.10   | 25.18    | 26.63   | 25.87   | 26.12    | 
| Sharpness     | 154.27 | 46.23 | 47.55 | 1.33    | 0.94    | 0.90     | 155.88  | 72.70   | 71.86    | 

**Wide-ResNet28-10**
|               | SGD     | SAM    | GSAM   | SGD + B | SAM + B | GSAM + B | SGD + C | SAM + C | GSAM + C | 
| :-----------: | :-----: | :----: | :----: | :-----: | :-----: | :------: | :-----: | :-----: | :------: | 
| Test Error(%) | 25.62   | 24.78  | 24.94  | 22.65   | 21.10   | 21.50    | 25.57   | 24.16   | 24.00    | 
| Sharpness     | 1113.26 | 456.17 | 435.17 | 22.72   | 10.99   | 12.37    | 1148.09 | 687.44  | 665.13   | 


### Algorithm Performance(t-imagenet)

**ResNet-18**
|               |    SGD |   SAM |   GSAM |   SGD + B |   SAM + B |   GSAM + B |   SGD + C |   SAM + C |   GSAM + C |
|:--------------|-------:|------:|-------:|----------:|----------:|-----------:|----------:|----------:|-----------:|
| Test Error(%) |  39.62 | 40.28 |  40.13 |     39.36 |     38.40 |      38.25 |     39.84 |     39.07 |      39.05 |
| Sharpness     | 279.07 | 97.52 |  97.23 |     13.33 |     10.65 |      10.31 |    333.73 |    183.59 |     192.29 |

**Wide-ResNet28-10**
|               |     SGD |    SAM |   GSAM |   SGD + B |   SAM + B |   GSAM + B |   SGD + C |   SAM + C |   GSAM + C |
|:--------------|--------:|-------:|-------:|----------:|----------:|-----------:|----------:|----------:|-----------:|
| Test Error(%) |   38.75 |  38.60 |  38.92 |     36.55 |     35.36 |      35.14 |     39.30 |     37.86 |      37.78 |
| Sharpness     | 2550.16 | 917.93 | 926.94 |    176.80 |    128.35 |     119.84 |   3066.71 |   1657.00 |    1803.57 |

### Algorithm Performance(imagenet)

**ResNet-50**
|               |    SGD |    SAM |   GSAM |   SGD + B |   SAM + B |   GSAM + B |   SGD + C |   SAM + C |   GSAM + C |
|:--------------|-------:|-------:|-------:|----------:|----------:|-----------:|----------:|----------:|-----------:|
| Test Error(%) |  32.23 |  30.62 |  30.63 |     29.19 |     29.06 |      29.23 |     30.58 |     29.96 |      29.72 |
| Sharpness     | 492.12 | 268.60 | 297.82 |     55.31 |     41.83 |      38.13 |   1504.53 |    853.04 |     670.51 |

## How to train models and measure sharpness

**train CNN-based models**
```python
python3 main_CNN.py　
```

**train ViT models**
```python
python3 main_ViT.py
```

**measure sharpness**
```python
cd my_sharpness
python3 eval_sharpness --model_path="{model_path}" --model="{model_path}"
```

## Acknowledgments
This code is highly based on [GSAM](https://github.com/juntang-zhuang/GSAM) , [sharpness-vs-generalization](https://github.com/tml-epfl/sharpness-vs-generalization) and [SPT_LSA_ViT](https://github.com/aanna0701/SPT_LSA_ViT). Thanks to the contributors of these projects.