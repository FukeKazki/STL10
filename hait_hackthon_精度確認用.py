import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision import models
import torch.nn as nn
from torch.autograd import Variable

model = models.vgg16(pretrained=True)
model.classifier[6] = nn.Linear(in_features=4096, out_features=10, bias=True)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

model.load_state_dict(torch.load('./model.pth', map_location=device))

classes = ('airplane', 'bird', 'car', 'cat', 'deer', 'dog', 'horse', 'monkey', 'ship', 'truck')

transform = transforms.Compose([
  transforms.Resize((96, 96)),
  transforms.ToTensor(),
  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

data_set = torchvision.datasets.ImageFolder(
  root='./images',
  transform=transform
)

test_loader = DataLoader(
    data_set, batch_size=1, shuffle=False,
    num_workers=4, drop_last=False,
)

model.eval()

correct = 0.0
total = 0.0
for data in test_loader:
    images, labels = data
    images = Variable(images)
    images = images.to(device)
    labels = labels.to(device)
    outputs = model(images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum()

print('Accuracy test images: %3f %%' % float(100 * correct / total))

class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))

for data in test_loader:
    images, labels = data
    images = images.to(device)
    labels = labels.to(device)
    # 検証
    outputs = model(images)
    # 正解のやつ
    _, predicted = torch.max(outputs, 1)
    is_correct = (predicted == labels)

    class_total[labels] += 1
    if is_correct:
      class_correct[labels] += 1

for i in range(10):
    print('Accuracy of %5s : %3f %%' % (classes[i], 100 * class_correct[i] / class_total[i]))