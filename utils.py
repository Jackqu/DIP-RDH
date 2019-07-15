import torch
import cv2
import numpy as np
def img_show(img, named_win = 'img' ):
    cv2.imshow(named_win, img)
    cv2.waitKey(0)

def tensor_to_numpy(img_t, normalize = 255.0):
    img_t = torch.squeeze(img_t,0)
    img_t = torch.squeeze(img_t,0)
    img_t = img_t * normalize
    img_numpy = img_t.detach().numpy()
    img_numpy = np.clip(img_numpy, 0, 255)
    return img_numpy



def generate_mask(mask_size, is_even = True):
    mask = torch.zeros(mask_size)
    assert len(mask.size()) == 4

    for i in range(mask.shape[2]):
        for j in range(mask.shape[3]):
            if is_even:
                if (i + j) % 2 == 0:
                    mask[:,:,i,j] = 1
            else:
                if (i + j) % 2 == 1:
                    mask[:,:,i,j] = 1
    return mask


def generate_histogram(img, predict, mask, hist_t = 3):
    """

    :param img:
    :param predict:
    :param mask:
    :param hist_t:
    :return:
    """
    diff = np.abs(img - predict)
    index = mask == 1
    diff = diff[index]
    hist = []
    for i in range(hist_t):
        hist_num = np.sum(diff == i)
        hist.append(hist_num)
    hist = np.array(hist)
    return hist



