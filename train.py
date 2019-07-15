import torch
import cv2
from models import UNet
import numpy as np
import utils
def train(net, input_noise, img, mask, optimizer, iter_num):
    criterion = torch.nn.MSELoss()
    for i in range(iter_num):
        optimizer.zero_grad()
        output = net(input_noise)
        loss = criterion(output * mask, img * mask)
        loss.backward()
        optimizer.step()
        print('train step {}, loss {}'.format(i, loss.item()))

        #imshow
        output_numpy = utils.tensor_to_numpy(output).astype(np.uint8)
        cv2.imshow('output', output_numpy)
        cv2.waitKey(1000)
        #histgram
        img_numpy = utils.tensor_to_numpy(img).astype(np.uint8)
        mask_numpy = utils.tensor_to_numpy(mask, normalize=1).astype(np.uint8)
        hist = utils.generate_histogram(img_numpy, output_numpy, mask_numpy, hist_t= 3)
        print('hist:', hist)




def main():
    #read image
    img = './imgs/lena.bmp'
    img = cv2.imread(img,0).astype(np.float32)
    print('img shape: ', img.shape)
    img = torch.from_numpy(img)
    img = torch.unsqueeze(img,0)
    img = torch.unsqueeze(img,0)
    img = img / 255.0
    #read net
    net = UNet()
    #get optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr = 0.001)
    #train parameter
    iter_num = 10000
    mask_size = (1,1,512,512)
    is_even = True
    mask = utils.generate_mask(mask_size, is_even)
    input_noise = torch.randn(*mask_size)
    #trian
    train(net, input_noise, img, mask, optimizer,iter_num)



if __name__ == '__main__':
    print('Start')
    main()
