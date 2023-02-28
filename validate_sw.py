import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse
import time

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Validate top-1 accuracy for PyTorch based Neural Network"
    )
    parser.add_argument(
        "--batchsize", help="number of samples for inference", type=int, default=100
    )
    parser.add_argument(
        "--model_path", help='path of the model file', required=True
    )
    parser.add_argument(
        "--dataset_root", help="dataset root dir for download/reuse", default="/tmp"
    )
    
    # define the model and import it's parameters from model_path
    class Net(nn.Module):
    
        def __init__(self):
            super(Net, self).__init__()
        
            # 2 convolution layers
            # 1 input image channel, 6 output channels, 5x5 convolution kernel
            self.conv1 = nn.Conv2d(3, 6, 5)
            # Adding pooling layer here only this time
            self.pool = nn.MaxPool2d(2,2)
            self.conv2 = nn.Conv2d(6, 16, 5)
        
            # 3 fully connected layers
            # an affine operation: y = Wx + b
            self.fc1 = nn.Linear(16 * 5 * 5, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, 10)
        
        
        def forward(self, x):
            # Convolution layers
            # x is the input image to the Neural Network
            # Max pooling over a 2x2 window (since it is square, it can also be specified as 2, instead of (2,2))
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
        
            # Now applying flattening before passing it to the Fully Connected layers
            # flatten all dimensions except the batch dimension
            x = torch.flatten(x, 1)
        
            # Now the Fully Connected layers
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
        
            return x
    
    args = parser.parse_args()
    net = Net()
    net.load_state_dict(torch.load(args.model_path))
    
    # Parsing other parameters
    bsize = args.batchsize
    dataset_root = args.dataset_root
    
    # Load the CIFAR-10 dataset
    from dataset_loading import cifar

    trainx, trainy, testx, testy, valx, valy = cifar.load_cifar_data(
        dataset_root, download=True, one_hot=False
    )
    
    # start recording time
    start_time = time.time()
    
    # Need to convert shape of dataset from (10000, 32, 32, 3) to (10000, 3, 32, 32)
    testx = np.moveaxis(testx, [3], [1])
    
    # Define the test image and label nparray
    test_imgs = testx
    test_labels = testy
    
    # Needed for benchmarking
    ok = 0 # number of correct predictions
    total = test_imgs.shape[0]
    
    # reshaping nparrays, further pre-processing and converting them to tensors for easy use with pytorch model
    test_labels = torch.tensor(test_labels)
    n_batches = int(total / bsize)
    test_imgs = test_imgs.reshape(n_batches, bsize, -1)
    test_labels = test_labels.reshape(n_batches, bsize)
    test_imgs = test_imgs.astype('float32')
    test_imgs = test_imgs / 255
    test_imgs = torch.tensor(test_imgs)
    
    # Not calculating gradients as we are not training
    with torch.no_grad():
        
        for i in range(n_batches):
         
            # need to reshape the data for each batch
            test_imgs_batch = test_imgs[i].reshape(bsize, 3, 32, 32)
            
            # Getting the ground truth
            exp = test_labels[i]
            
            # Getting the output and comapring it with ground truth
            outputs = net(test_imgs_batch)
            _, predicted = torch.max(outputs.data, 1)
            ok += (predicted == exp).sum().item()
            
    # stop recoring time
    end_time = time.time() - start_time
    print("Inference Time :  %s seconds" % (end_time))
    acc = 100.0 * ok / (total)
    print("Final accuracy: %f" % acc)