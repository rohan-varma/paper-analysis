import torch
import torchvision
from torchvision import transforms
from itertools import islice
# get CIFAR 10 data
training_data = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform = torchvision.transforms.ToTensor())
trainloader = torch.utils.data.DataLoader(training_data, batch_size=4,
                                          shuffle=True, num_workers=2)

# subscripting
print(type(training_data))
exit()
image, label = training_data[0]
# slicing doesn't seem to work
x = training_data[:10] # raises TypeError: Cannot handle this data type
# __getitem__() doesn't take slice object
x = training_data[0]
x = training_data[slice(0, 10)] # raises TypeError: Cannot handle this data type

# can work around with dataiter
dataiter = iter(trainloader)
batches = islice(dataiter, 0, 10)
