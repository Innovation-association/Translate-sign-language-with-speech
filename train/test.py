from paddlex.cls import transforms
import paddlex
import cv2


model_dir = './model/number/epoch_1'
test_jpg_path = './data/testData/number/saved_0001.jpg'  # 测试图片路径


train_transforms = transforms.Compose([
    transforms.RandomCrop(crop_size=224),
    transforms.Normalize()
])

model = paddlex.load_model(model_dir)
im = cv2.imread(test_jpg_path)
result = model.predict(im, topk=3, transforms=train_transforms)
print("Predict Result:", result)
