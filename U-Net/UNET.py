import torch
import os
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
from torch.autograd import Variable
import matplotlib as plt
from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader, is_image_file
import matplotlib.pyplot as plt
import random
from torch.utils.data.sampler import SubsetRandomSampler
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image



pixel_map = {(64, 128, 64): 0., (192, 0, 128): 1., (0, 128, 192): 2., (0, 128, 64): 3., (128, 0, 0): 4.,
             (64, 0, 128): 5., (64, 0, 192): 6., (192, 128, 64): 7., (192, 192, 128): 8., (64, 64, 128): 9.,
             (128, 0, 192): 10., (192, 0, 64): 11., (128, 128, 64): 12., (192, 0, 192): 13., (128, 64, 64): 14.,
             (64, 192, 128): 15., (64, 64, 0): 16., (128, 64, 128): 17., (128, 128, 192): 18., (0, 0, 192): 19.,
             (192, 128, 128): 20., (128, 128, 128): 21., (64, 128, 192): 22., (0, 0, 64): 23., (0, 64, 64): 24.,
             (192, 64, 128): 25., (128, 128, 0): 26., (192, 128, 192): 27., (64, 0, 64): 28., (192, 192, 0): 29.,
             (0, 0, 0): 30., (64, 192, 0): 31.}

# hyper parameters
learning_rate = 0.001
total_epoch = 30
batch_size = 4

test_length = 100
# load data set
torch.cuda.empty_cache()

def _make_data_set(dir):
    images = list()
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                item = path
                images.append(item)
    return images


class camvid_dataset(Dataset):
    def __init__(self, file_path_data, file_path_label, transform=None, label_transform=None):
        self.root_dir_data = file_path_data
        self.root_dir_label = file_path_label
        self.transforms = transform
        self.label_transforms = label_transform
        self.imgs = _make_data_set(file_path_data)
        self.labels = _make_data_set(file_path_label)

    def __getitem__(self, item):
        img_path = self.imgs[item]
        label_path = self.labels[item]
        img = default_loader(img_path)
        label = default_loader(label_path)

        pixels = label.load()
        final_labels = np.zeros((label.size[0], label.size[1]),dtype = np.long)
        for l in range(label.size[0]):
            for j in range(label.size[1]):
                final_labels[l, j] = pixel_map.get(pixels[l, j], 30.)
        final_labels = final_labels.T
        
        if self.transforms is not None:
            img = self.transforms(img)
        if self.label_transforms is not None:
            final_labels = self.label_transforms(final_labels)
        return img, torch.tensor(final_labels)

    def __len__(self):
        return len(self.imgs)


d_path = "/content/camvid/701_StillsRaw_full"
l_path = "/content/camvid/LabeledApproved_full"
trans = transforms.Compose([transforms.ToTensor()])





# Network class
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # first contracting layer
        base_channel_length = 32
        self.cont_l1_conv1 = nn.Conv2d(3, base_channel_length, 3, padding=1)
        self.cont_l1_conv2 = nn.Conv2d(base_channel_length, base_channel_length, 3, padding=1)
        # second contracting layer
        self.cont_l2_conv1 = nn.Conv2d(base_channel_length, base_channel_length * 2, 3, padding=1)
        self.cont_l2_conv2 = nn.Conv2d(2 * base_channel_length, 2 * base_channel_length, 3, padding=1)
        # third contracting layer
        self.cont_l3_conv1 = nn.Conv2d(2 * base_channel_length, 4 * base_channel_length, 3, padding=1)
        self.cont_l3_conv2 = nn.Conv2d(4 * base_channel_length, 4 * base_channel_length, 3, padding=1)
        # fourth contracting layer
        self.cont_l4_conv1 = nn.Conv2d(4 * base_channel_length, 8 * base_channel_length, 3, padding=1)
        self.cont_l4_conv2 = nn.Conv2d(8 * base_channel_length, 8 * base_channel_length, 3, padding=1)
        # fifth contracting layer
        self.cont_l5_conv1 = nn.Conv2d(8 * base_channel_length, 16 * base_channel_length, 3, padding=1)
        self.cont_l5_conv2 = nn.Conv2d(16 * base_channel_length, 16 * base_channel_length, 3, padding=1)

        # first up conv layer
        self.upconv_l1_conv1 = nn.Conv2d(16 * base_channel_length, 8 * base_channel_length, 1)
        # second up conv layer
        self.upconv_l2_conv1 = nn.Conv2d(8 * base_channel_length, 4 * base_channel_length, 1)
        # third up conv layer
        self.upconv_l3_conv1 = nn.Conv2d(4 * base_channel_length, 2 * base_channel_length, 1)
        # fourth up conv layer
        self.upconv_l4_conv1 = nn.Conv2d(2 * base_channel_length, base_channel_length, 1)

        # first expand layer
        self.exp_l1_conv1 = nn.Conv2d(16 * base_channel_length, 8 * base_channel_length, 3, padding=1)
        self.exp_l1_conv2 = nn.Conv2d(8 * base_channel_length, 8 * base_channel_length, 3, padding=1)
        # second expand layer
        self.exp_l2_conv1 = nn.Conv2d(8 * base_channel_length, 4 * base_channel_length, 3, padding=1)
        self.exp_l2_conv2 = nn.Conv2d(4 * base_channel_length, 4 * base_channel_length, 3, padding=1)
        # third expand layer
        self.exp_l3_conv1 = nn.Conv2d(4 * base_channel_length, 2 * base_channel_length, 3, padding=1)
        self.exp_l3_conv2 = nn.Conv2d(2 * base_channel_length, 2 * base_channel_length, 3, padding=1)
        # fourth expand layer
        self.exp_l4_conv1 = nn.Conv2d(2 * base_channel_length, base_channel_length, 3, padding=1)
        self.exp_l4_conv2 = nn.Conv2d(base_channel_length, base_channel_length, 3, padding=1)
        self.exp_l4_conv3 = nn.Conv2d(base_channel_length, 32, 1)

        # up sample layer
        self.upsample = nn.Upsample(mode='nearest', scale_factor=2)

    def crop(self, x, target):
        k, c, a, b = target.size()
        j, f, d, e = x.size()
        x1 = int(round((d - a) / 2.))
        x2 = int(round((e - b) / 2.))
        return x[:, :, x1:x1 + a, x2:x2 + b]

    def forward(self, x):
        c1 = F.relu(self.cont_l1_conv1(x), inplace=True)
        c1 = F.relu(self.cont_l1_conv2(c1))
        p1 = F.max_pool2d(c1, (2, 2))
        del x
        c2 = F.relu(self.cont_l2_conv1(p1))
        c2 = F.relu(self.cont_l2_conv2(c2))
        p2 = F.max_pool2d(c2, (2, 2))
        del p1
        c3 = F.relu(self.cont_l3_conv1(p2))
        c3 = F.relu(self.cont_l3_conv2(c3))
        p3 = F.max_pool2d(c3, (2, 2))
        del p2
        c4 = F.relu(self.cont_l4_conv1(p3))
        c4 = F.relu(self.cont_l4_conv2(c4))
        p4 = F.max_pool2d(c4, (2, 2))
        del p3
        c5 = F.relu(self.cont_l5_conv1(p4))
        c5 = F.relu(self.cont_l5_conv2(c5))
        del p4
        u1 = self.upconv_l1_conv1(c5)
        u1 = self.upsample(u1)
        del c5
        u2 = torch.cat((c4, u1), 1)
        u2 = F.relu(self.exp_l1_conv1(u2))
        u2 = F.relu(self.exp_l1_conv2(u2))
        del c4,u1
        u2 = self.upconv_l2_conv1(u2)
        u2 = self.upsample(u2)

        u3 = torch.cat((c3, u2), 1)
        u3 = F.relu(self.exp_l2_conv1(u3))
        u3 = F.relu(self.exp_l2_conv2(u3))
        u3 = self.upconv_l3_conv1(u3)
        u3 = self.upsample(u3)
        del c3, u2
        u4 = torch.cat((c2, u3), 1)
        u4 = F.relu(self.exp_l3_conv1(u4))
        u4 = F.relu(self.exp_l3_conv2(u4))
        del c2, u3
        u4 = self.upconv_l4_conv1(u4)
        u4 = self.upsample(u4)

        u5 = torch.cat((c1, u4), 1)
        u5 = F.relu(self.exp_l4_conv1(u5))
        u5 = F.relu(self.exp_l4_conv2(u5))
        output = F.relu(self.exp_l4_conv3(u5))
        del c1, u4, u5
        return output


unet = Net()

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(unet.parameters(), lr=learning_rate,amsgrad=True)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

unet.to(device)

mean = [0.41189489566336, 0.4251328133025, 0.4326707089857]
std = [0.27413549931506, 0.28506257482912, 0.28284674400252]

print(mean)
print(std)

trans2 = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])

my_dataset = camvid_dataset(d_path, l_path, trans2)

a = range(len(my_dataset))
test_index = random.sample(range(len(my_dataset)),test_length)
train_index = [x for x in a if x not in test_index]


train_sampler = SubsetRandomSampler(train_index)
test_sampler = SubsetRandomSampler(test_index)

       
train_loader = torch.utils.data.DataLoader(dataset=my_dataset, batch_size=batch_size,sampler= train_sampler, shuffle=False)
test_loader = torch.utils.data.DataLoader(dataset=my_dataset, batch_size=1,sampler= test_sampler, shuffle=False)

print((len(train_loader.dataset)-test_length))

count_step = 4
all_loss = list()
test_total_loss = list()
optimizer.zero_grad()
for epoch in range(total_epoch):
    this_epoch_loss = list()
    
    unet.train()
    optimizer.zero_grad()
    for i, (images, labels) in enumerate(train_loader):

        images, labels = images.to(device), labels.to(device)
        images = Variable(images)
        labels = Variable(labels)

        
        outputs = unet(images)
        
        
        

        loss = criterion(outputs, labels)/count_step
        loss.backward()
        if(i+1)%count_step == 0:
          optimizer.step()          
          optimizer.zero_grad()     
 
        max_val, max_arg = torch.max(outputs.data, dim=1, keepdim=True)
        
        
        
        
        
        

        print("this batch {} , loss is:{}".format(float(i), loss.item()))
        this_epoch_loss.append(loss.item())
    all_loss.append(np.mean(this_epoch_loss))
    print("epoch {} average loss was: {}".format(epoch, all_loss[-1]))
    
    unet.eval()
    test_loss = []
    for i, (images, labels) in enumerate(test_loader):
        images, labels = images.to(device), labels.to(device)
        images = Variable(images)
        labels = Variable(labels)
        
        outputs = unet(images)
        
        
        
        loss = criterion(outputs, labels)
        max_val, max_arg = torch.max(outputs.data, dim=1, keepdim=True)
        
        print("this batch for test {} , loss is:{}".format(float(i), loss.item()))
        test_loss.append(loss.item())
    print("epoch test {} average loss was: {}, accuracy = {}".format(epoch, np.mean(test_loss)))
    test_total_loss.append(np.mean(test_loss))







pixel_back_map = {0.:(64,128,64), 1.:(192,0,128), 2.:(0,128,192), 3.:(0,128,64), 4.:(128,0,0),
             5.:(64,0,128), 6.:(64,0,192), 7.:(192,128,64),8.:(192,192,128), 9.:(64,64,128),
             10.:(128,0,192), 11.:(192,0,64), 12.:(128,128,64), 13.:(192,0,192),14.:(128,64,64),
             15.:(64,192,128), 16.:(64,64,0),17.:(128,64,128), 18.:(128,128,192), 19.:(0,0,192),
             20.:(192,128,128), 21.:(128,128,128), 22.:(64,128,192), 23.:(0,0,64), 24.:(0,64,64),
             25.:(192,64,128), 26.:(128,128,0), 27.:(192,128,192), 28.:(64,0,64),29.:(192,192,0),
             30.:(0,0,0), 31.:(64,192,0)}
def create_image(input):
    print(input.shape)

    fig2 = plt.figure()
    img = Image.new('RGB', (960, 720))
    values = [[0 for t in range(960)] for u in range(720)]
    
    for i in range(720):
        for j in range(960):
            values[i][j] = pixel_back_map[input[i, j]]
    print(len(values))
    
    imgplot = plt.imshow(values)
    plt.show()



for i ,(images, labels) in enumerate(test_loader):
    unet.eval()
    torch.cuda.empty_cache()
    images = Variable(images)
    labels = Variable(labels)
    images = images.to(device)
    out = unet(images)
    print(images.shape)


    images = images.view(3,720,960)
    images = images.to('cpu')
    out = out.to('cpu')
    max_val, max_arg = torch.max(out.data, dim=1, keepdim=True)
    out = max_arg.view(720, 960)
    out = out.numpy()
    create_image(out)
    
    
    
    imgplot = plt.imshow(images.permute(1,2,0))
    plt.show()
    break



test_loss2 = [x/4 for x in test_total_loss]
fig = plt.figure()
plt.plot(all_loss,'r', label = 'train_loss')
plt.plot(test_loss2,'b', label = 'test_loss')
plt.legend(loc='upper right')
plt.title('average loss in epoch')
plt.xlabel('# of epoch')
plt.ylabel('average Loss')
plt.show()
