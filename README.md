


### **Create a Conda Environment**

```
conda create -n rt python=3.10
```

```
conda activate rt

conda install cudatoolkit==11.8 -c nvidia
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu118

pip install -r requirements.txt
pip install triton==2.3.1
pip install transformers==4.43.3
```

```
pip install timm==1.0.7 thop efficientnet_pytorch==0.7.1 einops grad-cam==1.5.4 dill==0.3.8 albumentations==1.4.11 pytorch_wavelets==1.3.0 tidecv PyWavelets opencv-python prettytable -i https://pypi.tuna.tsinghua.edu.cn/simple
```
