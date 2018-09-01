from __future__ import print_function
import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable #should be deprecated now
from apex.fp16_utils import to_python_float #not sure what for
from model import Net

#=====START: ADDED FOR DISTRIBUTED======
try:
    from apex.parallel import DistributedDataParallel as DDP 
except ImportError:
    raise ImportError("Please install apex from https://www.github.com/nvidia/apex to run this example.")

'''Import distributed data loader'''
import torch.utils.data
import torch.utils.data.distributed

'''Import torch.distributed'''
import torch.distributed as dist
#import torch.nn.parallel.DistributedDataParallel as DDP
#=====END:   ADDED FOR DISTRIBUTED======

parser = argparse.ArgumentParser(description='Pytorch MNIST Distributed Test on NGC')
parser.add_argument('--batch_size', type=int, default=64, metavar='N', 
                    help='input batch size for training (default: 64)')
parser.add_argument('--test_batch_size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no_cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log_interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')

#======START: ADDED FOR DISTRIBUTED======
'''
Add some distributed options. For explanation of dist-url and dist-backend please see
http://pytorch.org/tutorials/intermediate/dist_tuto.html

--local_rank will be supplied by the Pytorch launcher wrapper (torch.distributed.launch)
'''
parser.add_argument("--local_rank", default=0, type=int)
#=====END:   ADDED FOR DISTRIBUTED======

def main():
    global args 
    args = parser.parse_args()
    #SUBPROC_ID_STR = 'ProcID {}/{}: '.format(args.local_rank, args.world_size)
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        #print(SUBPROC_ID_STR + "Cuda device available; using cuda device for training")
        print("Cuda device available; using cuda device for training")
    #======START: ADDED FOR DISTRIBUTED======
    '''Add a convenience flag to see if we are running distributed'''
    args.distributed = False
    if 'WORLD_SIZE' in os.environ:
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.distributed = args.world_size > 1

    if args.distributed:
        '''Check that we are running with cuda, as distributed is only supported for cuda.'''
        assert args.cuda, "Distributed mode requires running with CUDA."
        # print("Training in distributed mode with {} total sub-processes".format)
        '''
        Set cuda device so everything is done on the right GPU.
        THIS MUST BE DONE AS SOON AS POSSIBLE.
        '''
        torch.cuda.set_device(args.local_rank)
        '''Initialize distributed communication'''
        torch.distributed.init_process_group(backend='nccl', init_method='env://')

    #=====END:   ADDED FOR DISTRIBUTED======
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    kwargs ={'num_workers':1, 'pin_memory':True} if args.cuda else {}


    #=====START: ADDED FOR DISTRIBUTED======
    '''
    Change sampler to distributed if running distributed.
    Shuffle data loader only if distributed.
    '''
    train_dataset = datasets.MNIST('./MNIST', train=True, download=True, 
                                    transform=transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.1307,), (0.3081,))
                                        ]))
    if args.distributed:
        #No need to provide rank to constructor because internally the constructor will all get_rank() in torch.distributed.
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    #divide batch size for each process to maintain overal batch size as given in args
    dist_batch_size = int(args.batch_size/dist.get_world_size()) if args.distributed else args.batch_size
    train_loader = torch.utils.data.DataLoader(train_dataset, sampler=train_sampler, 
                                                batch_size=dist_batch_size, shuffle=(train_sampler is None), **kwargs
                                                )
    #=====END:   ADDED FOR DISTRIBUTED======
    test_dataset = datasets.MNIST('./MNIST', train=False, transform=transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.1307,), (0.3081,))
                                    ]))
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.test_batch_size,
                                    shuffle=True, **kwargs)


    model = Net()
    if args.cuda:
        model = model.cuda()
    #=====START: ADDED FOR DISTRIBUTED======
    '''
    Wrap model in our version of DistributedDataParallel.
    This must be done AFTER the model is converted to cuda.
    '''

    if args.distributed:
        model = DDP(model)
    #=====END:   ADDED FOR DISTRIBUTED======

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    for epoch in range(1, args.epochs + 1):
        #=====START: ADDED FOR DISTRIBUTED======
        if args.distributed:
            train_sampler.set_epoch(epoch)
        #=====END:   ADDED FOR DISTRIBUTED======

        train(model, epoch, optimizer, train_loader)
        if args.local_rank == 0:
            test(model, test_loader)

def train(model, epoch, optimizer, train_loader):
    model.train()
    for idx, (data, target) in enumerate(train_loader, 1): #idx start from 1
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        #data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        out = model(data)
        loss = F.nll_loss(out, target)
        loss.backward()
        optimizer.step()
        # if idx % args.log_interval == 0 and args.local_rank == 0:
        #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        #         epoch, idx * len(data), len(train_loader.dataset),
        #         100. * idx / len(train_loader), loss.item()))
        if idx % args.log_interval == 0 and args.local_rank == 0:
            print('Rank {} ==> Train Epoch: {} [{}/{} ({} in total) ({:.0f}%)]'.format(
                args.local_rank, epoch, idx * len(data), len(train_loader.sampler), len(train_loader.dataset),
                100. * idx / len(train_loader)))
            if args.local_rank == 0:
                print('Rank {} ==> Loss: {:.6f}'.format(args.local_rank, loss.item()))

def test(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        with torch.no_grad():
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            out = model(data)
            loss = F.nll_loss(out, target, size_average=False)
            test_loss += loss.item()
            pred = torch.max(out, dim=1)[1]
            correct += pred.eq(target).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))




if __name__ == '__main__':
    main()





















