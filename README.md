# Densely-Self-guided-Wavelet-Network-for-Image-Denoising

Our network uses DWT and IDWT. Please install correspinding library as the following link: https://github.com/fbcotter/pytorch_wavelets

We have upload all the .py files and .txt file. Please unzip the training and valid data in the workspace as name_list.txt and val_gt.txt.

For test: 

Download the pre-trained model [Google drive](https://drive.google.com/file/d/18j1IFujKJEBCXaUQ4JdAwROxXKE1SRgX/view?usp=sharing)

Download testing data [Google drive](https://drive.google.com/file/d/1h-a2BfJbPV__1aDKgNJjpeqpXtrgFU_b/view?usp=sharing)

```bash
python submit.py
```

Please set parser.add_argument.use_ensemble as True when you test our model. The code will generate our ensemble results.

If you need to test the runtime, please change the parser.add_argument.use_ensemble in submit.py as False.

For train
```bash
python train.py
```


Reference:
[1] Liu Wei，Yan Qiong，Zhao Yuzhi. Densely Self-guided Wavelet Network for Image Denoising[C]. IEEE/CVF Conference on Computer Vision and  Pattern Recognition Workshops 2020 (CVPRW)

[2] S. Gu, Y. Li, L. V. Gool, and R. Timofte, "Self-Guided Network for Fast Image Denoising”

[3] P. Liu, H. Zhang, W. Lian, and W. Zuo, "Multi-level Wavelet Convolutional Neural Networks."

