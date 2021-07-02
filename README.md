# Duo-SegNet

This repo contains the supported pytorch code and configuration files to reproduce medical image segmentaion results of Duo-SegNet. 

![Dual View Architecture](img/duo_segnet.png?raw=true)

## Environment
Please prepare an environment with python=3.8, and then run the command "pip install -r requirements.txt" for the dependencies.

## Data Preparation

## Train/Test
- Train : Run the train script on nuclei dataset for 5% of labeled data. 
```bash
python train.py --dataset nuclei --ratio 0.05 --epoch 200
```

- Test : Run the test script on nuclei dataset. 
```bash
python test.py --dataset nuclei
```

## Reference

## Citing Duo-SegNet
