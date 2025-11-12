# SplitMeanFlow_MNIST
Test SplitMeanFlow and MeanFlow on MNIST dataset.

Paper: [SplitMeanFlow: Interval Splitting Consistency in Few-Step Generative Modeling](https://arxiv.org/abs/2507.16884)

## RUN
```bash
python train_splitmeanflow.py
python test.py --ckpt splitmeanflow_0070000.pt # generate samples
```

## RESULT
SplitMeanFlow and MeanFlow can generate samples in one step, FlowMatch can not.

![SplitMeanFlow Result](/splitmeanflow_result.png)
