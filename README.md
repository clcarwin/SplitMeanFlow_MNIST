# SplitMeanFlow_MNIST
Test SplitMeanFlow and MeanFlow on MNIST dataset.

## RUN
```bash
python train_splitmeanflow.py
python test.py --ckpt splitmeanflow_0070000.pt # generate samples
```

## RESULT
SplitMeanFlow and MeanFlow can generate samples in one step, FlowMatch can not.
![SplitMeanFlow Result](/splitmeanflow_result.png)