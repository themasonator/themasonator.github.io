# Classifying Images Of Skin Lesions Using Transfer Learning

## The Dataset

I used [this dataset](https://www.kaggle.com/datasets/salviohexia/isic-2019-skin-lesion-images-for-classification).

I will be creating a baseline image classifier based on this data in order to introduce myself to working on the following subjects: large datasets, image processing, pytorch, use of GPUs, transfer learning, neural classifiers, Amazon S3 and bias and variance.

## Pre-Processing

[I looked here to find what to do when the classes are unbalanced](https://machinelearningmastery.com/multi-class-imbalanced-classification/). Therefore I deleted all NV elements until there were as many elements in there as in the second largest class. If I had more GPU time to experiment with, I believe SMOTE oversampling could be useful here.

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

I followed [this guide](https://huggingface.co/docs/timm/models/tf-mixnet) to import the small mixnet model, which I chose due to its low compute needs relative to its performance.


```python
%pip install timm
import timm
model = timm.create_model('tf_mixnet_s', pretrained=True, num_classes=8)
model.eval()
```

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
