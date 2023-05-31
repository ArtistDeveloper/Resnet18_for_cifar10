import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import timm
import random
import numpy as np


if __name__ == '__main__':
    model_num = 2 # total number of models
    total_epoch = 100 # total epoch
    lr = 0.001 # initial learning rate # learning rate 0.01 -> 0.0001
    num_GPU = 1

    for s in range(model_num):
        # fix random seed
        seed_number = s
        random.seed(seed_number)
        np.random.seed(seed_number)
        torch.manual_seed(seed_number)

        # Active deterministic operation and avoid nondeterministic task(operation)
        torch.backends.cudnn.deterministic = True 
        torch.backends.cudnn.benchmark = False

        # Check if GPU is available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(device)

        # Define the data transforms
        transform_train = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(), # change [0, 255]int value to [0, 1] Float value(FloatTensor type)
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # 변경가능
        ])

        transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # 변경가능
        ])

        # Load the CIFAR-10 dataset
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=4 * num_GPU, pin_memory=True) # batch_size 128->256

        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=4 * num_GPU, pin_memory=True)

        # Define the ResNet-18 model with pre-trained weights
        model = timm.create_model('resnet18', pretrained=True, num_classes=10)
        model = model.to(device)  # Move the model to the GPU

        # Define the loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        # optimizer = optim.Adam(model.parameters(), lr=lr, momentum=0.9)
        optimizer = optim.Adam(model.parameters(), lr=lr)

        # Define the learning rate scheduler
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
        # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=0.01)

        def train():
            model.train()
            running_loss = 0.0
            
            for i, data in enumerate(trainloader, 0):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)  # Move the input data to the GPU
                # optimizer.zero_grad()
                for param in model.parameters():
                    param.grad = None

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                # running_loss += loss.item()
                running_loss += loss.detach()
                if i % 100 == 99:
                    print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 100))
                    running_loss = 0.0   
                    
        def test():
            model.eval()
            
            # Test the model
            correct = 0
            total = 0
            with torch.no_grad():
                for data in testloader:
                    images, labels = data
                    images, labels = images.to(device), labels.to(device)  # Move the input data to the GPU
                    outputs = model(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    # correct += (predicted == labels).sum().item()
                    correct += (predicted == labels).sum().detach()

            print('Accuracy of the network on the 10000 test images: %f %%' % (100 * correct / total))

        # Train the model
        for epoch in range(total_epoch):
            train()
            test()
            scheduler.step()

        print('Finished Training')

        # Save the checkpoint of the last model
        PATH = './resnet18_cifar10_%f_%d.pth' % (lr, seed_number)
        torch.save(model.state_dict(), PATH)