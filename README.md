# Duo-SegNet: Adversarial Dual-Views for Semi-Supervised Medical Image Segmentation

This repo contains the supported pytorch code and configuration files to reproduce medical image segmentaion results of [Duo-SegNet](https://arxiv.org/pdf/2108.11154.pdf). 

![Dual View Architecture](img/duo_segnet.png?raw=true)

<a href="https://www.codecogs.com/eqnedit.php?latex={\mathcal{F}_i(\cdot)}_{i=1}^2" target="_blank"><img src="https://latex.codecogs.com/gif.latex?{\mathcal{F}_i(\cdot)}_{i=1}^2" title="{\mathcal{F}_i(\cdot)}_{i=1}^2" /></a> and <a href="https://www.codecogs.com/eqnedit.php?latex=\psi&space;(\cdot)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\psi&space;(\cdot)" title="\psi (\cdot)" /></a> denote Segmentation networks and Critic network. Here, Critic criticizes between prediction masks and the ground truth masks to perform the min-max game.

## Environment
Please prepare an environment with python=3.8, and then run the command "pip install -r requirements.txt" for the dependencies.

## Data Preparation
- For experiments we used three datasets:
  - Nuclei (2018 Data Science Bowl)
  - Spleen (Medical segmentation decathlon - MSD)
  - Heart ([Medical segmentation decathlon - MSD)

- File structure
    ```
     data
      ├── nuclei
      |   ├── train
      │   │   ├── image
      │   │   │   └── 00ae65...
      │   │   └── mask
      │   │       └── 00ae65...       
      ├── spleen
      ├── heart
      │   
      |
     Duo-SegNet
      ├──train.py
      ...
    ```

 - Use [Med2Image](https://github.com/FNNDSC/med2image) to convert NIFTI to PNG.
 
## Train/Test
- Train : Run the train script on nuclei dataset for 5% of labeled data. 
```bash
python train.py --dataset nuclei --ratio 0.05 --epoch 200
```

- Test : Run the test script on nuclei dataset. 
```bash
python test.py --dataset nuclei
```

## Acknowledgements
This repository makes liberal use of code from [Deep Co-training](https://github.com/AlanChou/Deep-Co-Training-for-Semi-Supervised-Image-Recognition) and [pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)

## References
* [2018 Data Science Bowl](https://www.kaggle.com/c/data-science-bowl-2018)
* [Medical segmentation decathlon](http://medicaldecathlon.com/)

## Citing Duo-SegNet
```bash
    @inproceedings{peiris2021duo,
      title={Duo-SegNet: Adversarial Dual-Views for Semi-Supervised Medical Image Segmentation},
      author={Peiris, Himashi and Chen, Zhaolin and Egan, Gary and Harandi, Mehrtash},
      booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
      pages={428--438},
      year={2021},
      organization={Springer}
    }
```
