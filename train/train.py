from paddlex.cls import transforms
import os
import cv2
import numpy as np
import paddlex as pdx

# 数据集路径
base = './data/trainData/gestures'

train_list_path = './data/trainData/train_list.txt'      # 训练列表
labels_path = './data/trainData/number_labels.txt'       # 训练数据标签
test_jpg_path = './data/trainData/gestures/Standing/0000.jpg'  # 测试图片路径
model_save_dir = '.\\model\\gestures'                        # 模型存放路径

with open(os.path.join(train_list_path), 'w') as f:
    for i, cls_fold in enumerate(os.listdir(base)):
        cls_base = os.path.join(base, cls_fold)
        files = os.listdir(cls_base)
        print('{} train num:'.format(cls_fold), len(files))
        for pt in files:
            img = os.path.join(cls_fold, pt)
            info = img + ' ' + str(i) + '\n'
            f.write(info)

with open(os.path.join(labels_path), 'w') as f:
    for i, cls_fold in enumerate(os.listdir(base)):
        f.write(cls_fold+'\n')

train_transforms = transforms.Compose([
    transforms.RandomCrop(crop_size=224),
    transforms.Normalize()
])

train_dataset = pdx.datasets.ImageNet(
    data_dir=base,
    file_list=train_list_path,
    label_list=labels_path,
    transforms=train_transforms,
    shuffle=True)

num_classes = len(train_dataset.labels)
model = pdx.cls.ResNet18(num_classes=num_classes)
model.train(num_epochs=100,             # 总训练轮数
            train_dataset=train_dataset,
            train_batch_size=32,        # 一次训练所选取的样本数
            lr_decay_epochs=[5, 10, 15],
            learning_rate=2e-2,         # 学习率
            save_dir=model_save_dir,    # 模型存放路径
            log_interval_steps=5,       # 输出日志间隔
            save_interval_epochs=4)     # 存储频率

im = cv2.imread(test_jpg_path)
result = model.predict(test_jpg_path, topk=1, transforms=train_transforms)
print("Predict Result:", result)