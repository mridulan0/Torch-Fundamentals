import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

def dataset_dataloader():
    # transforms.Compose() == puts together a bunch of transformations
    # transforms.ToTensor() == converts loaded images into PyTorch tensors
    # transorms.Normalize() == adjusts values of the tensor so that their average is zero and standard deviation is 1
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))])

    # root == the path where the data will be stored
    # train == whether we want to use the dataset for training
    # download == whether we want to download it or not if it already hasn't been downloaded
    # transform == the transformations that we want to apply to the images
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)

    # organizes the input data into batches
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                            shuffle=True, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    return trainset, trainloader, classes

def imshow(img):
    # unnormalize
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))

    
if __name__ == '__main__':
    trainset, trainloader, classes = dataset_dataloader()
    dataiter = iter(trainloader)
    images, labels = next(dataiter)

    imshow(torchvision.utils.make_grid(images))
    print(' '.join('%5s' % classes[labels[j]] for j in range(4)))