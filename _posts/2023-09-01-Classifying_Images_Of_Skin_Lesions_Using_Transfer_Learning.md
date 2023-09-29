# Classifying Images Of Skin Lesions Using Transfer Learning

## The Dataset

I used this dataset: https://www.kaggle.com/datasets/salviohexia/isic-2019-skin-lesion-images-for-classification.

I will be creating a baseline image classifier based on this data in order to introduce myself to working on the following subjects: large datasets, image processing, pytorch, use of GPUs, transfer learning, neural classifiers, Amazon S3 and bias and variance.

## Pre-Processing

I looked here to find what to do when the classes are unbalanced: https://machinelearningmastery.com/multi-class-imbalanced-classification/ Therefore I deleted all NV elements until there were as many elements in there as in the second largest class. If I had more GPU time to experiment with, I believe SMOTE oversampling could be useful here.

As the test dataset must match the distribution, I will sample 10% for the train and dev sets from each class.


```python
def copyListOver(bucket, category, className, listOfNames):
    key = 'path/' + category + '/' + className
    for name in listOfNames:
        copy_source = {
            'Bucket': bucket,
            'Key': skindata + className + '/' + name
        }
        print(key)
        print(skindata + className + '/')
        conn.copy(copy_source, bucket, key + '/' + name)

```


```python
from numpy import asarray
from numpy import save
from tensorflow.keras.utils import load_img
from tensorflow.keras.utils import img_to_array
import random
allClasses = ["AK", "BCC", "BKL", "DF", "MEL", "NV", "SCC", "VASC"]
photos = list()
for className in allClasses:
    subfolder = skindata + className
    paginator = conn.get_paginator('list_objects')
    operation_parameters = {'Bucket': bucket,
                        'Prefix': subfolder}
    contents = conn.list_objects_v2(Bucket=bucket, Prefix=subfolder)['Contents']
    page_iterator = paginator.paginate(**operation_parameters)
    allFiles = []
    for page in page_iterator:
        for f in page['Contents']:
            allFiles.append(f['Key'].rsplit('/', 1)[1])
        
    random.shuffle(allFiles)
    chunk_size = len(allFiles)//10
    test = allFiles[:chunk_size]
    copyListOver(bucket, 'test', className, test)
    dev = allFiles[chunk_size:(chunk_size*2)]
    copyListOver(bucket, 'dev', className, dev)
    train = allFiles[(chunk_size*2):]
    copyListOver(bucket, 'train', className, train)
```


```python
from numpy import asarray
from numpy import save
from tensorflow.keras.utils import load_img
from tensorflow.keras.utils import img_to_array
import random
s3 = boto3.resource('s3')
path = "path/train/NV/"
bucket = s3.Bucket("bucket-name")
objects = list(bucket.objects.filter(Prefix=path))
num_objects_to_delete = int(len(objects) * (1-(4522/12875)))
objects_to_delete = objects_to_delete = random.sample(objects, num_objects_to_delete)
for obj in objects_to_delete:
    obj.delete()
```

## The Model

I followed this guide to import the small mixnet model, which I chose due to its low compute needs relative to its performance: https://huggingface.co/docs/timm/models/tf-mixnet.


```python
%pip install timm
import timm
model = timm.create_model('tf_mixnet_s', pretrained=True, num_classes=8)
model.eval()
```
<div overflow:scroll; height:150px>
    Collecting timm
      Obtaining dependency information for timm from https://files.pythonhosted.org/packages/7a/bd/2c56be7a3b5bc71cf85a405246b89d5359f942c9f7fb6db6306d9d056092/timm-0.9.7-py3-none-any.whl.metadata
      Downloading timm-0.9.7-py3-none-any.whl.metadata (58 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m58.8/58.8 kB[0m [31m9.2 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting torch>=1.7 (from timm)
      Downloading torch-2.0.1-cp310-cp310-manylinux1_x86_64.whl (619.9 MB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m619.9/619.9 MB[0m [31m1.7 MB/s[0m eta [36m0:00:00[0m00:01[0m00:01[0m
    [?25hCollecting torchvision (from timm)
      Downloading torchvision-0.15.2-cp310-cp310-manylinux1_x86_64.whl (6.0 MB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m6.0/6.0 MB[0m [31m103.5 MB/s[0m eta [36m0:00:00[0m00:01[0m
    [?25hRequirement already satisfied: pyyaml in /home/ec2-user/anaconda3/envs/tensorflow2_p310/lib/python3.10/site-packages (from timm) (6.0)
    Collecting huggingface-hub (from timm)
      Obtaining dependency information for huggingface-hub from https://files.pythonhosted.org/packages/aa/f3/3fc97336a0e90516901befd4f500f08d691034d387406fdbde85bea827cc/huggingface_hub-0.17.3-py3-none-any.whl.metadata
      Downloading huggingface_hub-0.17.3-py3-none-any.whl.metadata (13 kB)
    Collecting safetensors (from timm)
      Obtaining dependency information for safetensors from https://files.pythonhosted.org/packages/6c/f0/c17bbdb1e5f9dab29d44cade445135789f75f8f08ea2728d04493ea8412b/safetensors-0.3.3-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata
      Downloading safetensors-0.3.3-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (4.7 kB)
    Requirement already satisfied: filelock in /home/ec2-user/anaconda3/envs/tensorflow2_p310/lib/python3.10/site-packages (from torch>=1.7->timm) (3.12.2)
    Requirement already satisfied: typing-extensions in /home/ec2-user/anaconda3/envs/tensorflow2_p310/lib/python3.10/site-packages (from torch>=1.7->timm) (4.5.0)
    Requirement already satisfied: sympy in /home/ec2-user/anaconda3/envs/tensorflow2_p310/lib/python3.10/site-packages (from torch>=1.7->timm) (1.12)
    Requirement already satisfied: networkx in /home/ec2-user/anaconda3/envs/tensorflow2_p310/lib/python3.10/site-packages (from torch>=1.7->timm) (3.1)
    Requirement already satisfied: jinja2 in /home/ec2-user/anaconda3/envs/tensorflow2_p310/lib/python3.10/site-packages (from torch>=1.7->timm) (3.1.2)
    Collecting nvidia-cuda-nvrtc-cu11==11.7.99 (from torch>=1.7->timm)
      Downloading nvidia_cuda_nvrtc_cu11-11.7.99-2-py3-none-manylinux1_x86_64.whl (21.0 MB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m21.0/21.0 MB[0m [31m18.3 MB/s[0m eta [36m0:00:00[0m00:01[0m00:01[0m
    [?25hCollecting nvidia-cuda-runtime-cu11==11.7.99 (from torch>=1.7->timm)
      Downloading nvidia_cuda_runtime_cu11-11.7.99-py3-none-manylinux1_x86_64.whl (849 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m849.3/849.3 kB[0m [31m8.0 MB/s[0m eta [36m0:00:00[0m00:01[0m
    [?25hCollecting nvidia-cuda-cupti-cu11==11.7.101 (from torch>=1.7->timm)
      Downloading nvidia_cuda_cupti_cu11-11.7.101-py3-none-manylinux1_x86_64.whl (11.8 MB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m11.8/11.8 MB[0m [31m26.6 MB/s[0m eta [36m0:00:00[0m00:01[0m00:01[0m
    [?25hCollecting nvidia-cudnn-cu11==8.5.0.96 (from torch>=1.7->timm)
      Downloading nvidia_cudnn_cu11-8.5.0.96-2-py3-none-manylinux1_x86_64.whl (557.1 MB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m557.1/557.1 MB[0m [31m1.0 MB/s[0m eta [36m0:00:00[0m00:01[0m00:01[0m
    [?25hCollecting nvidia-cublas-cu11==11.10.3.66 (from torch>=1.7->timm)
      Downloading nvidia_cublas_cu11-11.10.3.66-py3-none-manylinux1_x86_64.whl (317.1 MB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m317.1/317.1 MB[0m [31m1.4 MB/s[0m eta [36m0:00:00[0m00:01[0m00:01[0m
    [?25hCollecting nvidia-cufft-cu11==10.9.0.58 (from torch>=1.7->timm)
      Downloading nvidia_cufft_cu11-10.9.0.58-py3-none-manylinux1_x86_64.whl (168.4 MB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m168.4/168.4 MB[0m [31m2.2 MB/s[0m eta [36m0:00:00[0m00:01[0m00:01[0m
    [?25hCollecting nvidia-curand-cu11==10.2.10.91 (from torch>=1.7->timm)
      Downloading nvidia_curand_cu11-10.2.10.91-py3-none-manylinux1_x86_64.whl (54.6 MB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m54.6/54.6 MB[0m [31m3.0 MB/s[0m eta [36m0:00:00[0m:00:01[0m00:01[0m
    [?25hCollecting nvidia-cusolver-cu11==11.4.0.1 (from torch>=1.7->timm)
      Downloading nvidia_cusolver_cu11-11.4.0.1-2-py3-none-manylinux1_x86_64.whl (102.6 MB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m102.6/102.6 MB[0m [31m3.7 MB/s[0m eta [36m0:00:00[0m00:01[0m00:01[0m
    [?25hCollecting nvidia-cusparse-cu11==11.7.4.91 (from torch>=1.7->timm)
      Downloading nvidia_cusparse_cu11-11.7.4.91-py3-none-manylinux1_x86_64.whl (173.2 MB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m173.2/173.2 MB[0m [31m2.2 MB/s[0m eta [36m0:00:00[0m00:01[0m00:01[0m
    [?25hCollecting nvidia-nccl-cu11==2.14.3 (from torch>=1.7->timm)
      Downloading nvidia_nccl_cu11-2.14.3-py3-none-manylinux1_x86_64.whl (177.1 MB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m177.1/177.1 MB[0m [31m6.7 MB/s[0m eta [36m0:00:00[0m00:01[0m00:01[0m
    [?25hCollecting nvidia-nvtx-cu11==11.7.91 (from torch>=1.7->timm)
      Downloading nvidia_nvtx_cu11-11.7.91-py3-none-manylinux1_x86_64.whl (98 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m98.6/98.6 kB[0m [31m18.7 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting triton==2.0.0 (from torch>=1.7->timm)
      Downloading triton-2.0.0-1-cp310-cp310-manylinux2014_x86_64.manylinux_2_17_x86_64.whl (63.3 MB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m63.3/63.3 MB[0m [31m9.3 MB/s[0m eta [36m0:00:00[0m:00:01[0m00:01[0m
    [?25hRequirement already satisfied: setuptools in /home/ec2-user/anaconda3/envs/tensorflow2_p310/lib/python3.10/site-packages (from nvidia-cublas-cu11==11.10.3.66->torch>=1.7->timm) (68.0.0)
    Requirement already satisfied: wheel in /home/ec2-user/anaconda3/envs/tensorflow2_p310/lib/python3.10/site-packages (from nvidia-cublas-cu11==11.10.3.66->torch>=1.7->timm) (0.40.0)
    Collecting cmake (from triton==2.0.0->torch>=1.7->timm)
      Obtaining dependency information for cmake from https://files.pythonhosted.org/packages/de/94/cba4b3ddc0d4555967cce8fd6e9fbced98a6bf62857db71c2400a7b6e183/cmake-3.27.5-py2.py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata
      Downloading cmake-3.27.5-py2.py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata (6.7 kB)
    Collecting lit (from triton==2.0.0->torch>=1.7->timm)
      Downloading lit-17.0.1.tar.gz (154 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m154.7/154.7 kB[0m [31m2.3 MB/s[0m eta [36m0:00:00[0m00:01[0m
    [?25h  Installing build dependencies ... [?25ldone
    [?25h  Getting requirements to build wheel ... [?25ldone
    [?25h  Installing backend dependencies ... [?25ldone
    [?25h  Preparing metadata (pyproject.toml) ... [?25ldone
    [?25hRequirement already satisfied: fsspec in /home/ec2-user/anaconda3/envs/tensorflow2_p310/lib/python3.10/site-packages (from huggingface-hub->timm) (2023.6.0)
    Requirement already satisfied: requests in /home/ec2-user/anaconda3/envs/tensorflow2_p310/lib/python3.10/site-packages (from huggingface-hub->timm) (2.31.0)
    Requirement already satisfied: tqdm>=4.42.1 in /home/ec2-user/anaconda3/envs/tensorflow2_p310/lib/python3.10/site-packages (from huggingface-hub->timm) (4.65.0)
    Requirement already satisfied: packaging>=20.9 in /home/ec2-user/anaconda3/envs/tensorflow2_p310/lib/python3.10/site-packages (from huggingface-hub->timm) (21.3)
    Requirement already satisfied: numpy in /home/ec2-user/anaconda3/envs/tensorflow2_p310/lib/python3.10/site-packages (from torchvision->timm) (1.25.1)
    Requirement already satisfied: pillow!=8.3.*,>=5.3.0 in /home/ec2-user/anaconda3/envs/tensorflow2_p310/lib/python3.10/site-packages (from torchvision->timm) (9.4.0)
    Requirement already satisfied: pyparsing!=3.0.5,>=2.0.2 in /home/ec2-user/anaconda3/envs/tensorflow2_p310/lib/python3.10/site-packages (from packaging>=20.9->huggingface-hub->timm) (3.0.9)
    Requirement already satisfied: MarkupSafe>=2.0 in /home/ec2-user/anaconda3/envs/tensorflow2_p310/lib/python3.10/site-packages (from jinja2->torch>=1.7->timm) (2.1.3)
    Requirement already satisfied: charset-normalizer<4,>=2 in /home/ec2-user/anaconda3/envs/tensorflow2_p310/lib/python3.10/site-packages (from requests->huggingface-hub->timm) (3.2.0)
    Requirement already satisfied: idna<4,>=2.5 in /home/ec2-user/anaconda3/envs/tensorflow2_p310/lib/python3.10/site-packages (from requests->huggingface-hub->timm) (3.4)
    Requirement already satisfied: urllib3<3,>=1.21.1 in /home/ec2-user/anaconda3/envs/tensorflow2_p310/lib/python3.10/site-packages (from requests->huggingface-hub->timm) (1.26.14)
    Requirement already satisfied: certifi>=2017.4.17 in /home/ec2-user/anaconda3/envs/tensorflow2_p310/lib/python3.10/site-packages (from requests->huggingface-hub->timm) (2023.5.7)
    Requirement already satisfied: mpmath>=0.19 in /home/ec2-user/anaconda3/envs/tensorflow2_p310/lib/python3.10/site-packages (from sympy->torch>=1.7->timm) (1.3.0)
    Downloading timm-0.9.7-py3-none-any.whl (2.2 MB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m2.2/2.2 MB[0m [31m19.5 MB/s[0m eta [36m0:00:00[0m:00:01[0m
    [?25hDownloading huggingface_hub-0.17.3-py3-none-any.whl (295 kB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m295.0/295.0 kB[0m [31m2.6 MB/s[0m eta [36m0:00:00[0mta [36m0:00:01[0m
    [?25hDownloading safetensors-0.3.3-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (1.3 MB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m1.3/1.3 MB[0m [31m5.4 MB/s[0m eta [36m0:00:00[0m0:00:01[0m
    [?25hDownloading cmake-3.27.5-py2.py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl (26.1 MB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m26.1/26.1 MB[0m [31m10.3 MB/s[0m eta [36m0:00:00[0m:00:01[0m00:01[0m
    [?25hBuilding wheels for collected packages: lit
      Building wheel for lit (pyproject.toml) ... [?25ldone
    [?25h  Created wheel for lit: filename=lit-17.0.1-py3-none-any.whl size=93254 sha256=47a39fc8d64e0a0f4d3513c7f4af916131a6b941854cf98a24f7b0a631c8fed5
      Stored in directory: /home/ec2-user/.cache/pip/wheels/cf/3a/a0/f65551951357f983270eb3b210b98c6be543f3ed5cf89deba4
    Successfully built lit
    Installing collected packages: safetensors, lit, cmake, nvidia-nvtx-cu11, nvidia-nccl-cu11, nvidia-cusparse-cu11, nvidia-curand-cu11, nvidia-cufft-cu11, nvidia-cuda-runtime-cu11, nvidia-cuda-nvrtc-cu11, nvidia-cuda-cupti-cu11, nvidia-cublas-cu11, nvidia-cusolver-cu11, nvidia-cudnn-cu11, huggingface-hub, triton, torch, torchvision, timm
    Successfully installed cmake-3.27.5 huggingface-hub-0.17.3 lit-17.0.1 nvidia-cublas-cu11-11.10.3.66 nvidia-cuda-cupti-cu11-11.7.101 nvidia-cuda-nvrtc-cu11-11.7.99 nvidia-cuda-runtime-cu11-11.7.99 nvidia-cudnn-cu11-8.5.0.96 nvidia-cufft-cu11-10.9.0.58 nvidia-curand-cu11-10.2.10.91 nvidia-cusolver-cu11-11.4.0.1 nvidia-cusparse-cu11-11.7.4.91 nvidia-nccl-cu11-2.14.3 nvidia-nvtx-cu11-11.7.91 safetensors-0.3.3 timm-0.9.7 torch-2.0.1 torchvision-0.15.2 triton-2.0.0
    Note: you may need to restart the kernel to use updated packages.



    Downloading model.safetensors:   0%|          | 0.00/16.7M [00:00<?, ?B/s]





    EfficientNet(
      (conv_stem): Conv2dSame(3, 16, kernel_size=(3, 3), stride=(2, 2), bias=False)
      (bn1): BatchNormAct2d(
        16, eps=0.001, momentum=0.1, affine=True, track_running_stats=True
        (drop): Identity()
        (act): ReLU(inplace=True)
      )
      (blocks): Sequential(
        (0): Sequential(
          (0): DepthwiseSeparableConv(
            (conv_dw): Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=16, bias=False)
            (bn1): BatchNormAct2d(
              16, eps=0.001, momentum=0.1, affine=True, track_running_stats=True
              (drop): Identity()
              (act): ReLU(inplace=True)
            )
            (se): Identity()
            (conv_pw): Conv2d(16, 16, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn2): BatchNormAct2d(
              16, eps=0.001, momentum=0.1, affine=True, track_running_stats=True
              (drop): Identity()
              (act): Identity()
            )
            (drop_path): Identity()
          )
        )
        (1): Sequential(
          (0): InvertedResidual(
            (conv_pw): MixedConv2d(
              (0): Conv2d(8, 48, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (1): Conv2d(8, 48, kernel_size=(1, 1), stride=(1, 1), bias=False)
            )
            (bn1): BatchNormAct2d(
              96, eps=0.001, momentum=0.1, affine=True, track_running_stats=True
              (drop): Identity()
              (act): ReLU(inplace=True)
            )
            (conv_dw): Conv2dSame(96, 96, kernel_size=(3, 3), stride=(2, 2), groups=96, bias=False)
            (bn2): BatchNormAct2d(
              96, eps=0.001, momentum=0.1, affine=True, track_running_stats=True
              (drop): Identity()
              (act): ReLU(inplace=True)
            )
            (se): Identity()
            (conv_pwl): MixedConv2d(
              (0): Conv2d(48, 12, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (1): Conv2d(48, 12, kernel_size=(1, 1), stride=(1, 1), bias=False)
            )
            (bn3): BatchNormAct2d(
              24, eps=0.001, momentum=0.1, affine=True, track_running_stats=True
              (drop): Identity()
              (act): Identity()
            )
            (drop_path): Identity()
          )
          (1): InvertedResidual(
            (conv_pw): MixedConv2d(
              (0): Conv2d(12, 36, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (1): Conv2d(12, 36, kernel_size=(1, 1), stride=(1, 1), bias=False)
            )
            (bn1): BatchNormAct2d(
              72, eps=0.001, momentum=0.1, affine=True, track_running_stats=True
              (drop): Identity()
              (act): ReLU(inplace=True)
            )
            (conv_dw): Conv2d(72, 72, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=72, bias=False)
            (bn2): BatchNormAct2d(
              72, eps=0.001, momentum=0.1, affine=True, track_running_stats=True
              (drop): Identity()
              (act): ReLU(inplace=True)
            )
            (se): Identity()
            (conv_pwl): MixedConv2d(
              (0): Conv2d(36, 12, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (1): Conv2d(36, 12, kernel_size=(1, 1), stride=(1, 1), bias=False)
            )
            (bn3): BatchNormAct2d(
              24, eps=0.001, momentum=0.1, affine=True, track_running_stats=True
              (drop): Identity()
              (act): Identity()
            )
            (drop_path): Identity()
          )
        )
        (2): Sequential(
          (0): InvertedResidual(
            (conv_pw): Conv2d(24, 144, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn1): BatchNormAct2d(
              144, eps=0.001, momentum=0.1, affine=True, track_running_stats=True
              (drop): Identity()
              (act): SiLU(inplace=True)
            )
            (conv_dw): MixedConv2d(
              (0): Conv2dSame(48, 48, kernel_size=(3, 3), stride=(2, 2), groups=48, bias=False)
              (1): Conv2dSame(48, 48, kernel_size=(5, 5), stride=(2, 2), groups=48, bias=False)
              (2): Conv2dSame(48, 48, kernel_size=(7, 7), stride=(2, 2), groups=48, bias=False)
            )
            (bn2): BatchNormAct2d(
              144, eps=0.001, momentum=0.1, affine=True, track_running_stats=True
              (drop): Identity()
              (act): SiLU(inplace=True)
            )
            (se): SqueezeExcite(
              (conv_reduce): Conv2d(144, 12, kernel_size=(1, 1), stride=(1, 1))
              (act1): SiLU(inplace=True)
              (conv_expand): Conv2d(12, 144, kernel_size=(1, 1), stride=(1, 1))
              (gate): Sigmoid()
            )
            (conv_pwl): Conv2d(144, 40, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn3): BatchNormAct2d(
              40, eps=0.001, momentum=0.1, affine=True, track_running_stats=True
              (drop): Identity()
              (act): Identity()
            )
            (drop_path): Identity()
          )
          (1): InvertedResidual(
            (conv_pw): MixedConv2d(
              (0): Conv2d(20, 120, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (1): Conv2d(20, 120, kernel_size=(1, 1), stride=(1, 1), bias=False)
            )
            (bn1): BatchNormAct2d(
              240, eps=0.001, momentum=0.1, affine=True, track_running_stats=True
              (drop): Identity()
              (act): SiLU(inplace=True)
            )
            (conv_dw): MixedConv2d(
              (0): Conv2d(120, 120, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=120, bias=False)
              (1): Conv2d(120, 120, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=120, bias=False)
            )
            (bn2): BatchNormAct2d(
              240, eps=0.001, momentum=0.1, affine=True, track_running_stats=True
              (drop): Identity()
              (act): SiLU(inplace=True)
            )
            (se): SqueezeExcite(
              (conv_reduce): Conv2d(240, 20, kernel_size=(1, 1), stride=(1, 1))
              (act1): SiLU(inplace=True)
              (conv_expand): Conv2d(20, 240, kernel_size=(1, 1), stride=(1, 1))
              (gate): Sigmoid()
            )
            (conv_pwl): MixedConv2d(
              (0): Conv2d(120, 20, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (1): Conv2d(120, 20, kernel_size=(1, 1), stride=(1, 1), bias=False)
            )
            (bn3): BatchNormAct2d(
              40, eps=0.001, momentum=0.1, affine=True, track_running_stats=True
              (drop): Identity()
              (act): Identity()
            )
            (drop_path): Identity()
          )
          (2): InvertedResidual(
            (conv_pw): MixedConv2d(
              (0): Conv2d(20, 120, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (1): Conv2d(20, 120, kernel_size=(1, 1), stride=(1, 1), bias=False)
            )
            (bn1): BatchNormAct2d(
              240, eps=0.001, momentum=0.1, affine=True, track_running_stats=True
              (drop): Identity()
              (act): SiLU(inplace=True)
            )
            (conv_dw): MixedConv2d(
              (0): Conv2d(120, 120, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=120, bias=False)
              (1): Conv2d(120, 120, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=120, bias=False)
            )
            (bn2): BatchNormAct2d(
              240, eps=0.001, momentum=0.1, affine=True, track_running_stats=True
              (drop): Identity()
              (act): SiLU(inplace=True)
            )
            (se): SqueezeExcite(
              (conv_reduce): Conv2d(240, 20, kernel_size=(1, 1), stride=(1, 1))
              (act1): SiLU(inplace=True)
              (conv_expand): Conv2d(20, 240, kernel_size=(1, 1), stride=(1, 1))
              (gate): Sigmoid()
            )
            (conv_pwl): MixedConv2d(
              (0): Conv2d(120, 20, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (1): Conv2d(120, 20, kernel_size=(1, 1), stride=(1, 1), bias=False)
            )
            (bn3): BatchNormAct2d(
              40, eps=0.001, momentum=0.1, affine=True, track_running_stats=True
              (drop): Identity()
              (act): Identity()
            )
            (drop_path): Identity()
          )
          (3): InvertedResidual(
            (conv_pw): MixedConv2d(
              (0): Conv2d(20, 120, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (1): Conv2d(20, 120, kernel_size=(1, 1), stride=(1, 1), bias=False)
            )
            (bn1): BatchNormAct2d(
              240, eps=0.001, momentum=0.1, affine=True, track_running_stats=True
              (drop): Identity()
              (act): SiLU(inplace=True)
            )
            (conv_dw): MixedConv2d(
              (0): Conv2d(120, 120, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=120, bias=False)
              (1): Conv2d(120, 120, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=120, bias=False)
            )
            (bn2): BatchNormAct2d(
              240, eps=0.001, momentum=0.1, affine=True, track_running_stats=True
              (drop): Identity()
              (act): SiLU(inplace=True)
            )
            (se): SqueezeExcite(
              (conv_reduce): Conv2d(240, 20, kernel_size=(1, 1), stride=(1, 1))
              (act1): SiLU(inplace=True)
              (conv_expand): Conv2d(20, 240, kernel_size=(1, 1), stride=(1, 1))
              (gate): Sigmoid()
            )
            (conv_pwl): MixedConv2d(
              (0): Conv2d(120, 20, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (1): Conv2d(120, 20, kernel_size=(1, 1), stride=(1, 1), bias=False)
            )
            (bn3): BatchNormAct2d(
              40, eps=0.001, momentum=0.1, affine=True, track_running_stats=True
              (drop): Identity()
              (act): Identity()
            )
            (drop_path): Identity()
          )
        )
        (3): Sequential(
          (0): InvertedResidual(
            (conv_pw): Conv2d(40, 240, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn1): BatchNormAct2d(
              240, eps=0.001, momentum=0.1, affine=True, track_running_stats=True
              (drop): Identity()
              (act): SiLU(inplace=True)
            )
            (conv_dw): MixedConv2d(
              (0): Conv2dSame(80, 80, kernel_size=(3, 3), stride=(2, 2), groups=80, bias=False)
              (1): Conv2dSame(80, 80, kernel_size=(5, 5), stride=(2, 2), groups=80, bias=False)
              (2): Conv2dSame(80, 80, kernel_size=(7, 7), stride=(2, 2), groups=80, bias=False)
            )
            (bn2): BatchNormAct2d(
              240, eps=0.001, momentum=0.1, affine=True, track_running_stats=True
              (drop): Identity()
              (act): SiLU(inplace=True)
            )
            (se): SqueezeExcite(
              (conv_reduce): Conv2d(240, 10, kernel_size=(1, 1), stride=(1, 1))
              (act1): SiLU(inplace=True)
              (conv_expand): Conv2d(10, 240, kernel_size=(1, 1), stride=(1, 1))
              (gate): Sigmoid()
            )
            (conv_pwl): MixedConv2d(
              (0): Conv2d(120, 40, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (1): Conv2d(120, 40, kernel_size=(1, 1), stride=(1, 1), bias=False)
            )
            (bn3): BatchNormAct2d(
              80, eps=0.001, momentum=0.1, affine=True, track_running_stats=True
              (drop): Identity()
              (act): Identity()
            )
            (drop_path): Identity()
          )
          (1): InvertedResidual(
            (conv_pw): Conv2d(80, 480, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn1): BatchNormAct2d(
              480, eps=0.001, momentum=0.1, affine=True, track_running_stats=True
              (drop): Identity()
              (act): SiLU(inplace=True)
            )
            (conv_dw): MixedConv2d(
              (0): Conv2d(240, 240, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=240, bias=False)
              (1): Conv2d(240, 240, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=240, bias=False)
            )
            (bn2): BatchNormAct2d(
              480, eps=0.001, momentum=0.1, affine=True, track_running_stats=True
              (drop): Identity()
              (act): SiLU(inplace=True)
            )
            (se): SqueezeExcite(
              (conv_reduce): Conv2d(480, 20, kernel_size=(1, 1), stride=(1, 1))
              (act1): SiLU(inplace=True)
              (conv_expand): Conv2d(20, 480, kernel_size=(1, 1), stride=(1, 1))
              (gate): Sigmoid()
            )
            (conv_pwl): MixedConv2d(
              (0): Conv2d(240, 40, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (1): Conv2d(240, 40, kernel_size=(1, 1), stride=(1, 1), bias=False)
            )
            (bn3): BatchNormAct2d(
              80, eps=0.001, momentum=0.1, affine=True, track_running_stats=True
              (drop): Identity()
              (act): Identity()
            )
            (drop_path): Identity()
          )
          (2): InvertedResidual(
            (conv_pw): Conv2d(80, 480, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn1): BatchNormAct2d(
              480, eps=0.001, momentum=0.1, affine=True, track_running_stats=True
              (drop): Identity()
              (act): SiLU(inplace=True)
            )
            (conv_dw): MixedConv2d(
              (0): Conv2d(240, 240, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=240, bias=False)
              (1): Conv2d(240, 240, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=240, bias=False)
            )
            (bn2): BatchNormAct2d(
              480, eps=0.001, momentum=0.1, affine=True, track_running_stats=True
              (drop): Identity()
              (act): SiLU(inplace=True)
            )
            (se): SqueezeExcite(
              (conv_reduce): Conv2d(480, 20, kernel_size=(1, 1), stride=(1, 1))
              (act1): SiLU(inplace=True)
              (conv_expand): Conv2d(20, 480, kernel_size=(1, 1), stride=(1, 1))
              (gate): Sigmoid()
            )
            (conv_pwl): MixedConv2d(
              (0): Conv2d(240, 40, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (1): Conv2d(240, 40, kernel_size=(1, 1), stride=(1, 1), bias=False)
            )
            (bn3): BatchNormAct2d(
              80, eps=0.001, momentum=0.1, affine=True, track_running_stats=True
              (drop): Identity()
              (act): Identity()
            )
            (drop_path): Identity()
          )
        )
        (4): Sequential(
          (0): InvertedResidual(
            (conv_pw): MixedConv2d(
              (0): Conv2d(40, 240, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (1): Conv2d(40, 240, kernel_size=(1, 1), stride=(1, 1), bias=False)
            )
            (bn1): BatchNormAct2d(
              480, eps=0.001, momentum=0.1, affine=True, track_running_stats=True
              (drop): Identity()
              (act): SiLU(inplace=True)
            )
            (conv_dw): MixedConv2d(
              (0): Conv2d(160, 160, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=160, bias=False)
              (1): Conv2d(160, 160, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=160, bias=False)
              (2): Conv2d(160, 160, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), groups=160, bias=False)
            )
            (bn2): BatchNormAct2d(
              480, eps=0.001, momentum=0.1, affine=True, track_running_stats=True
              (drop): Identity()
              (act): SiLU(inplace=True)
            )
            (se): SqueezeExcite(
              (conv_reduce): Conv2d(480, 40, kernel_size=(1, 1), stride=(1, 1))
              (act1): SiLU(inplace=True)
              (conv_expand): Conv2d(40, 480, kernel_size=(1, 1), stride=(1, 1))
              (gate): Sigmoid()
            )
            (conv_pwl): MixedConv2d(
              (0): Conv2d(240, 60, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (1): Conv2d(240, 60, kernel_size=(1, 1), stride=(1, 1), bias=False)
            )
            (bn3): BatchNormAct2d(
              120, eps=0.001, momentum=0.1, affine=True, track_running_stats=True
              (drop): Identity()
              (act): Identity()
            )
            (drop_path): Identity()
          )
          (1): InvertedResidual(
            (conv_pw): MixedConv2d(
              (0): Conv2d(60, 180, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (1): Conv2d(60, 180, kernel_size=(1, 1), stride=(1, 1), bias=False)
            )
            (bn1): BatchNormAct2d(
              360, eps=0.001, momentum=0.1, affine=True, track_running_stats=True
              (drop): Identity()
              (act): SiLU(inplace=True)
            )
            (conv_dw): MixedConv2d(
              (0): Conv2d(90, 90, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=90, bias=False)
              (1): Conv2d(90, 90, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=90, bias=False)
              (2): Conv2d(90, 90, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), groups=90, bias=False)
              (3): Conv2d(90, 90, kernel_size=(9, 9), stride=(1, 1), padding=(4, 4), groups=90, bias=False)
            )
            (bn2): BatchNormAct2d(
              360, eps=0.001, momentum=0.1, affine=True, track_running_stats=True
              (drop): Identity()
              (act): SiLU(inplace=True)
            )
            (se): SqueezeExcite(
              (conv_reduce): Conv2d(360, 60, kernel_size=(1, 1), stride=(1, 1))
              (act1): SiLU(inplace=True)
              (conv_expand): Conv2d(60, 360, kernel_size=(1, 1), stride=(1, 1))
              (gate): Sigmoid()
            )
            (conv_pwl): MixedConv2d(
              (0): Conv2d(180, 60, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (1): Conv2d(180, 60, kernel_size=(1, 1), stride=(1, 1), bias=False)
            )
            (bn3): BatchNormAct2d(
              120, eps=0.001, momentum=0.1, affine=True, track_running_stats=True
              (drop): Identity()
              (act): Identity()
            )
            (drop_path): Identity()
          )
          (2): InvertedResidual(
            (conv_pw): MixedConv2d(
              (0): Conv2d(60, 180, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (1): Conv2d(60, 180, kernel_size=(1, 1), stride=(1, 1), bias=False)
            )
            (bn1): BatchNormAct2d(
              360, eps=0.001, momentum=0.1, affine=True, track_running_stats=True
              (drop): Identity()
              (act): SiLU(inplace=True)
            )
            (conv_dw): MixedConv2d(
              (0): Conv2d(90, 90, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=90, bias=False)
              (1): Conv2d(90, 90, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=90, bias=False)
              (2): Conv2d(90, 90, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), groups=90, bias=False)
              (3): Conv2d(90, 90, kernel_size=(9, 9), stride=(1, 1), padding=(4, 4), groups=90, bias=False)
            )
            (bn2): BatchNormAct2d(
              360, eps=0.001, momentum=0.1, affine=True, track_running_stats=True
              (drop): Identity()
              (act): SiLU(inplace=True)
            )
            (se): SqueezeExcite(
              (conv_reduce): Conv2d(360, 60, kernel_size=(1, 1), stride=(1, 1))
              (act1): SiLU(inplace=True)
              (conv_expand): Conv2d(60, 360, kernel_size=(1, 1), stride=(1, 1))
              (gate): Sigmoid()
            )
            (conv_pwl): MixedConv2d(
              (0): Conv2d(180, 60, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (1): Conv2d(180, 60, kernel_size=(1, 1), stride=(1, 1), bias=False)
            )
            (bn3): BatchNormAct2d(
              120, eps=0.001, momentum=0.1, affine=True, track_running_stats=True
              (drop): Identity()
              (act): Identity()
            )
            (drop_path): Identity()
          )
        )
        (5): Sequential(
          (0): InvertedResidual(
            (conv_pw): Conv2d(120, 720, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn1): BatchNormAct2d(
              720, eps=0.001, momentum=0.1, affine=True, track_running_stats=True
              (drop): Identity()
              (act): SiLU(inplace=True)
            )
            (conv_dw): MixedConv2d(
              (0): Conv2dSame(144, 144, kernel_size=(3, 3), stride=(2, 2), groups=144, bias=False)
              (1): Conv2dSame(144, 144, kernel_size=(5, 5), stride=(2, 2), groups=144, bias=False)
              (2): Conv2dSame(144, 144, kernel_size=(7, 7), stride=(2, 2), groups=144, bias=False)
              (3): Conv2dSame(144, 144, kernel_size=(9, 9), stride=(2, 2), groups=144, bias=False)
              (4): Conv2dSame(144, 144, kernel_size=(11, 11), stride=(2, 2), groups=144, bias=False)
            )
            (bn2): BatchNormAct2d(
              720, eps=0.001, momentum=0.1, affine=True, track_running_stats=True
              (drop): Identity()
              (act): SiLU(inplace=True)
            )
            (se): SqueezeExcite(
              (conv_reduce): Conv2d(720, 60, kernel_size=(1, 1), stride=(1, 1))
              (act1): SiLU(inplace=True)
              (conv_expand): Conv2d(60, 720, kernel_size=(1, 1), stride=(1, 1))
              (gate): Sigmoid()
            )
            (conv_pwl): Conv2d(720, 200, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn3): BatchNormAct2d(
              200, eps=0.001, momentum=0.1, affine=True, track_running_stats=True
              (drop): Identity()
              (act): Identity()
            )
            (drop_path): Identity()
          )
          (1): InvertedResidual(
            (conv_pw): Conv2d(200, 1200, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn1): BatchNormAct2d(
              1200, eps=0.001, momentum=0.1, affine=True, track_running_stats=True
              (drop): Identity()
              (act): SiLU(inplace=True)
            )
            (conv_dw): MixedConv2d(
              (0): Conv2d(300, 300, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=300, bias=False)
              (1): Conv2d(300, 300, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=300, bias=False)
              (2): Conv2d(300, 300, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), groups=300, bias=False)
              (3): Conv2d(300, 300, kernel_size=(9, 9), stride=(1, 1), padding=(4, 4), groups=300, bias=False)
            )
            (bn2): BatchNormAct2d(
              1200, eps=0.001, momentum=0.1, affine=True, track_running_stats=True
              (drop): Identity()
              (act): SiLU(inplace=True)
            )
            (se): SqueezeExcite(
              (conv_reduce): Conv2d(1200, 100, kernel_size=(1, 1), stride=(1, 1))
              (act1): SiLU(inplace=True)
              (conv_expand): Conv2d(100, 1200, kernel_size=(1, 1), stride=(1, 1))
              (gate): Sigmoid()
            )
            (conv_pwl): MixedConv2d(
              (0): Conv2d(600, 100, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (1): Conv2d(600, 100, kernel_size=(1, 1), stride=(1, 1), bias=False)
            )
            (bn3): BatchNormAct2d(
              200, eps=0.001, momentum=0.1, affine=True, track_running_stats=True
              (drop): Identity()
              (act): Identity()
            )
            (drop_path): Identity()
          )
          (2): InvertedResidual(
            (conv_pw): Conv2d(200, 1200, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn1): BatchNormAct2d(
              1200, eps=0.001, momentum=0.1, affine=True, track_running_stats=True
              (drop): Identity()
              (act): SiLU(inplace=True)
            )
            (conv_dw): MixedConv2d(
              (0): Conv2d(300, 300, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=300, bias=False)
              (1): Conv2d(300, 300, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=300, bias=False)
              (2): Conv2d(300, 300, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), groups=300, bias=False)
              (3): Conv2d(300, 300, kernel_size=(9, 9), stride=(1, 1), padding=(4, 4), groups=300, bias=False)
            )
            (bn2): BatchNormAct2d(
              1200, eps=0.001, momentum=0.1, affine=True, track_running_stats=True
              (drop): Identity()
              (act): SiLU(inplace=True)
            )
            (se): SqueezeExcite(
              (conv_reduce): Conv2d(1200, 100, kernel_size=(1, 1), stride=(1, 1))
              (act1): SiLU(inplace=True)
              (conv_expand): Conv2d(100, 1200, kernel_size=(1, 1), stride=(1, 1))
              (gate): Sigmoid()
            )
            (conv_pwl): MixedConv2d(
              (0): Conv2d(600, 100, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (1): Conv2d(600, 100, kernel_size=(1, 1), stride=(1, 1), bias=False)
            )
            (bn3): BatchNormAct2d(
              200, eps=0.001, momentum=0.1, affine=True, track_running_stats=True
              (drop): Identity()
              (act): Identity()
            )
            (drop_path): Identity()
          )
        )
      )
      (conv_head): Conv2d(200, 1536, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn2): BatchNormAct2d(
        1536, eps=0.001, momentum=0.1, affine=True, track_running_stats=True
        (drop): Identity()
        (act): ReLU(inplace=True)
      )
      (global_pool): SelectAdaptivePool2d (pool_type=avg, flatten=Flatten(start_dim=1, end_dim=-1))
      (classifier): Linear(in_features=1536, out_features=8, bias=True)
    )
    
<\div>
```python
from PIL import Image
from io import BytesIO
import torch
from timm.data.transforms_factory import create_transform
from timm.data import resolve_data_config
from numpy import asarray
from numpy import save
import random
import boto3
from torch.utils.data import Dataset, DataLoader
```

## DataSet and DataLoader

To work in pytorch, I needed to use the DataSet and DataLoader classes to use the train/test/dev split. Here I loaded each already preprocessed image into memory since I had a lot of memory available. On an instance with less memory I would have loaded only the JPEG files into memory as they take up less space.


```python
class SkinLesionDataset(Dataset):
    def __init__(self, directory):
        self.labelTextAndOutputs = {
            "AK" : torch.FloatTensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), 
            "BCC" : torch.FloatTensor([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), 
            "BKL" : torch.FloatTensor([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]), 
            "DF" : torch.FloatTensor([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]), 
            "MEL" : torch.FloatTensor([0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]), 
            "NV" : torch.FloatTensor([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]), 
            "SCC" : torch.FloatTensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]), 
            "VASC" : torch.FloatTensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
        }
        self.images = []
        self.labels = []
        config = resolve_data_config({}, model=model)
        self.transform = create_transform(**config)

        self.s3_client = boto3.client('s3')
        self.bucket_name = "bucket-name"
        image_key = directory

        allClasses = ["AK", "BCC", "BKL", "DF", "MEL", "NV", "SCC", "VASC"]
        photos = list()
        for className in allClasses:
            subfolder = directory + className
            paginator = self.s3_client.get_paginator('list_objects')
            operation_parameters = {'Bucket': self.bucket_name,
                                'Prefix': subfolder}
    
            contents = self.s3_client.list_objects_v2(Bucket=self.bucket_name, Prefix=subfolder)['Contents']
            page_iterator = paginator.paginate(**operation_parameters)
            for page in page_iterator:
                for f in page['Contents']:
                    if(f['Key'][-1] == 'g'):
                        response = self.s3_client.get_object(Bucket=self.bucket_name, Key=f['Key'])
                        image_data = response['Body'].read()
                        image = Image.open(BytesIO(image_data)).convert("RGB")
                        tensor = self.transform(image)
                        self.images.append(tensor)
                        label_text = f['Key'].rsplit('/', 2)[-2]
                        self.labels.append(self.labelTextAndOutputs[label_text])
        
    def __getitem__(self, index):
        return self.images[index], self.labels[index]
        
    def __len__(self):
        return len(self.images)
        
```


```python
batch_size = 128
dev_dataset = SkinLesionDataset("path/dev/")
dev_data_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle = True)
train_dataset = SkinLesionDataset("path/train/")
train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle = True)
test_dataset = SkinLesionDataset("path/test/")
test_data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle = True)
```

## Hyperparameters

I used cross entropy loss as I am training a classifier model.

I am using the adam optimizer since it tunes the learning rate as it runs. I could further investigate its performance relative to other optimizers if I had more GPU time.


```python
import torch.nn as nn
import torch.optim as optim
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
```

## Transfer Learning

To do transfer learning, I am freezing all layers beside the very last, the classifier layer. Given more compute availability I could have added an extra convolutional layer.


```python
layer = 0
for child in model.children():
    layer += 1
    if layer < 7:
        for param in child.parameters():
            param.requires_grad = False
    else:
        for param in child.parameters():
            param.requires_grad = True
```

## Model Training

Here I chose to train for 50 epochs since that is when the model stopped significantly improving in dev set accuracy.


```python
num_epochs = 20
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_data_loader:
        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss / len(train_data_loader)}")

    model.eval()
    correct_predictions = 0
    total_samples = 0
    with torch.no_grad():
        for images, labels in dev_data_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            _, labelled = torch.max(labels, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labelled).sum().item()

    accuracy = correct_predictions / total_samples
    print(f"Dev Accuracy: {accuracy * 100:.2f}%")
```

    Epoch 1/20, Loss: 2.7907642980791487
    Dev Accuracy: 38.19%
    Epoch 2/20, Loss: 2.0174669398451752
    Dev Accuracy: 44.64%
    Epoch 3/20, Loss: 1.7409650469725986
    Dev Accuracy: 47.17%
    Epoch 4/20, Loss: 1.5933564462751713
    Dev Accuracy: 48.37%
    Epoch 5/20, Loss: 1.4786861707579415
    Dev Accuracy: 49.40%
    Epoch 6/20, Loss: 1.4062994729797795
    Dev Accuracy: 50.30%
    Epoch 7/20, Loss: 1.3349409497009133
    Dev Accuracy: 50.24%
    Epoch 8/20, Loss: 1.275085615099601
    Dev Accuracy: 51.63%
    Epoch 9/20, Loss: 1.2315531845362682
    Dev Accuracy: 52.29%
    Epoch 10/20, Loss: 1.1824219333675672
    Dev Accuracy: 52.77%
    Epoch 11/20, Loss: 1.1545079777825553
    Dev Accuracy: 54.40%
    Epoch 12/20, Loss: 1.1112331478100903
    Dev Accuracy: 53.31%
    Epoch 13/20, Loss: 1.083980179620239
    Dev Accuracy: 55.48%
    Epoch 14/20, Loss: 1.0533887451549746
    Dev Accuracy: 54.40%
    Epoch 15/20, Loss: 1.03462549144367
    Dev Accuracy: 55.00%
    Epoch 16/20, Loss: 1.0099426378618996
    Dev Accuracy: 55.90%
    Epoch 17/20, Loss: 0.9905995124915861
    Dev Accuracy: 55.42%
    Epoch 18/20, Loss: 0.964174489367683
    Dev Accuracy: 55.84%
    Epoch 19/20, Loss: 0.953206720779527
    Dev Accuracy: 56.27%
    Epoch 20/20, Loss: 0.9280581873542858
    Dev Accuracy: 55.60%



```python
torch.save(model.state_dict(), 'model_weights.pth')
```


```python
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_data_loader:
        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss / len(train_data_loader)}")

    model.eval()
    correct_predictions = 0
    total_samples = 0
    with torch.no_grad():
        for images, labels in dev_data_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            _, labelled = torch.max(labels, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labelled).sum().item()

    accuracy = correct_predictions / total_samples
    print(f"Dev Accuracy: {accuracy * 100:.2f}%")
```

    Epoch 1/10, Loss: 0.9184121467032522
    Dev Accuracy: 56.93%
    Epoch 2/10, Loss: 0.9009082396075411
    Dev Accuracy: 56.93%
    Epoch 3/10, Loss: 0.8853725161192552
    Dev Accuracy: 56.75%
    Epoch 4/10, Loss: 0.8700846495493403
    Dev Accuracy: 57.17%
    Epoch 5/10, Loss: 0.8674838706007544
    Dev Accuracy: 57.59%
    Epoch 6/10, Loss: 0.853587952987203
    Dev Accuracy: 57.83%
    Epoch 7/10, Loss: 0.8492388095495835
    Dev Accuracy: 57.95%
    Epoch 8/10, Loss: 0.8361938275256247
    Dev Accuracy: 57.29%
    Epoch 9/10, Loss: 0.8259401754388269
    Dev Accuracy: 57.71%
    Epoch 10/10, Loss: 0.8160843866051368
    Dev Accuracy: 57.89%



```python
torch.save(model.state_dict(), 'model_weights_after_30.pth')
```


```python
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_data_loader:
        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss / len(train_data_loader)}")

    model.eval()
    correct_predictions = 0
    total_samples = 0
    with torch.no_grad():
        for images, labels in dev_data_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            _, labelled = torch.max(labels, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labelled).sum().item()

    accuracy = correct_predictions / total_samples
    print(f"Dev Accuracy: {accuracy * 100:.2f}%")
```

    Epoch 1/10, Loss: 0.8146244720467981
    Dev Accuracy: 58.01%
    Epoch 2/10, Loss: 0.8022170252395127
    Dev Accuracy: 58.25%
    Epoch 3/10, Loss: 0.7955482416557815
    Dev Accuracy: 58.61%
    Epoch 4/10, Loss: 0.7849635498703651
    Dev Accuracy: 58.92%
    Epoch 5/10, Loss: 0.7758960161568984
    Dev Accuracy: 59.04%
    Epoch 6/10, Loss: 0.7832610460947145
    Dev Accuracy: 58.13%
    Epoch 7/10, Loss: 0.7819349658939073
    Dev Accuracy: 58.37%
    Epoch 8/10, Loss: 0.7683895938801315
    Dev Accuracy: 58.67%
    Epoch 9/10, Loss: 0.756737546538407
    Dev Accuracy: 58.49%
    Epoch 10/10, Loss: 0.7594883700586715
    Dev Accuracy: 58.55%



```python
torch.save(model.state_dict(), 'model_weights_after_40.pth')
```


```python
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_data_loader:
        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss / len(train_data_loader)}")

    model.eval()
    correct_predictions = 0
    total_samples = 0
    with torch.no_grad():
        for images, labels in dev_data_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            _, labelled = torch.max(labels, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labelled).sum().item()

    accuracy = correct_predictions / total_samples
    print(f"Dev Accuracy: {accuracy * 100:.2f}%")
```

    Epoch 1/10, Loss: 0.7518191286977732
    Dev Accuracy: 58.86%
    Epoch 2/10, Loss: 0.7513745286554661
    Dev Accuracy: 59.10%
    Epoch 3/10, Loss: 0.74187171459198
    Dev Accuracy: 58.86%
    Epoch 4/10, Loss: 0.7366870037227307
    Dev Accuracy: 59.34%
    Epoch 5/10, Loss: 0.7353763608437665
    Dev Accuracy: 59.82%
    Epoch 6/10, Loss: 0.731105275873868
    Dev Accuracy: 59.22%
    Epoch 7/10, Loss: 0.7302301762238989
    Dev Accuracy: 58.92%
    Epoch 8/10, Loss: 0.7257057765744767
    Dev Accuracy: 58.80%
    Epoch 9/10, Loss: 0.7225250253137553
    Dev Accuracy: 59.34%
    Epoch 10/10, Loss: 0.7233179346570429
    Dev Accuracy: 58.80%



```python
torch.save(model.state_dict(), 'model_weights_after_50.pth')
```

## Evaluation

I calculated the test and train set accuracy here to properly evaluate the model.


```python
model.load_state_dict(torch.load('model_weights_after_50.pth'))
```




    <All keys matched successfully>




```python
model.eval()
correct_predictions = 0
total_samples = 0
with torch.no_grad():
    for images, labels in train_data_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        _, labelled = torch.max(labels, 1)
        total_samples += labels.size(0)
        correct_predictions += (predicted == labelled).sum().item()
accuracy = correct_predictions / total_samples
print(f"Train Accuracy: {accuracy * 100:.2f}%")
```

    Train Accuracy: 75.94%



```python
model.eval()
correct_predictions = 0
total_samples = 0
with torch.no_grad():
    for images, labels in test_data_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        _, labelled = torch.max(labels, 1)
        total_samples += labels.size(0)
        correct_predictions += (predicted == labelled).sum().item()
accuracy = correct_predictions / total_samples
print(f"Test Accuracy: {accuracy * 100:.2f}%")
```

    Test Accuracy: 63.36%


The test set accuracy is in fact higher than the dev set accuracy, suggesting I the results also follow for the test set.

If I assumed that human-level performance were 90% accuracy, then I would calculate the bias this way:
90 - 75.94 = 14.06%
and the variance this way:
75.94 - 58.80 = 17.14%
or perhaps this way:
75.94 - 63.36 = 12.58%

This indicates that the bias and variance are very similar and to improving the model I would prioritise reducing the bias. To do that I would use a bigger convolutional neural network or a transformer model, of which many are available as this model was selected primarily for its small size. If it hadn't already seemingly converged I could also train the model for even longer.
