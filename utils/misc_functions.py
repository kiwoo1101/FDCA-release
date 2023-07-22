"""
Created on Thu Oct 21 11:09:09 2017

@author: Utku Ozbulak - github.com/utkuozbulak
"""
import os
import copy
import numpy as np
from PIL import Image
import matplotlib.cm as mpl_color_map
from matplotlib.colors import ListedColormap
from matplotlib import pyplot as plt

import torch
from torch.autograd import Variable
from torchvision import models
import PIL.ImageOps


def convert_to_grayscale(im_as_arr):
    """
        Converts 3d image to grayscale

    Args:
        im_as_arr (numpy arr): RGB image with shape (D,W,H)

    returns:
        grayscale_im (numpy_arr): Grayscale image with shape (1,W,D)
    """
    grayscale_im = np.sum(np.abs(im_as_arr), axis=0)
    im_max = np.percentile(grayscale_im, 99)
    im_min = np.min(grayscale_im)
    grayscale_im = (np.clip((grayscale_im - im_min) / (im_max - im_min), 0, 1))
    grayscale_im = np.expand_dims(grayscale_im, axis=0)
    return grayscale_im


def save_gradient_images(gradient, file_name):
    """
        Exports the original gradient image

    Args:
        gradient (np arr): Numpy array of the gradient with shape (3, 224, 224)
        file_name (str): File name to be exported
    """
    if not os.path.exists('../results'):
        os.makedirs('../results')
    # Normalize
    gradient = gradient - gradient.min()
    gradient /= gradient.max()
    # Save image
    path_to_file = os.path.join('../results', file_name + '.png')
    save_image(gradient, path_to_file)


def save_image_all(image_list, height_size=256, width_size=128, padding=1, save_path='../test.png'):
    dir_path = os.path.dirname(save_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    if isinstance(image_list, list):
        if isinstance(image_list[0], list):
            pass
        else:
            image_list = [image_list]
    else:
        image_list = [[image_list]]
    # height_size = 256
    # width_size = 128
    row_num = len(image_list)
    col_num = len(image_list[0])
    # 第一个参数RGB表示创建RGB彩色图，第二个参数传入元组指定图片大小，第三个参数可指定颜色，默认为黑色
    target = Image.new('RGB',
                       ((width_size + padding) * col_num - padding, (height_size + padding) * row_num - padding),
                       color=(255, 255, 255))  # 创建成品图的画布
    for row in range(row_num):
        for col in range(col_num):
            if isinstance(image_list[row][col], (np.ndarray, torch.Tensor)):
                image_numpy = image_list[row][col] if isinstance(image_list[row][col], np.ndarray) else \
                    image_list[row][col].cpu().numpy()
                image_numpy = (np.transpose(image_numpy, (1, 2, 0)) if image_numpy.shape[0] == 3 else
                               image_numpy)
                im = Image.fromarray(np.uint8(image_numpy))
            else:
                im = image_list[row][col]
            target.paste(im, (0 + (width_size + padding) * col, 0 + (height_size + padding) * row))
    SAVE_QUALITY = 95  # 1（最差）到95（最佳）。 默认值为75，使用中应尽量避免高于95的值
    # PIL.ImageOps.invert(target).save(save_path, quality=SAVE_QUALITY)  # 成品图保存
    target.save(save_path, quality=SAVE_QUALITY)  # 成品图保存


def save_image_single_row(image_list, height_size=256, width_size=128, padding=2, padding2=15, save_path='../test1.png'):
    dir_path = os.path.dirname(save_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    if isinstance(image_list, list):
        if isinstance(image_list[0], list):
            pass
        else:
            image_list = [image_list]
    else:
        image_list = [[image_list]]
    # height_size = 256
    # width_size = 128
    row_num = len(image_list)
    col_num = len(image_list[0])
    # 第一个参数RGB表示创建RGB彩色图，第二个参数传入元组指定图片大小，第三个参数可指定颜色，默认为黑色
    wid_length=((width_size + padding) * col_num - padding+padding2)*row_num-padding2
    target = Image.new('RGB',
                       (wid_length, height_size),
                       color=(255, 255, 255))  # 创建成品图的画布
    for row in range(row_num):
        for col in range(col_num):
            if isinstance(image_list[row][col], (np.ndarray, torch.Tensor)):
                image_numpy = image_list[row][col] if isinstance(image_list[row][col], np.ndarray) else \
                    image_list[row][col].cpu().numpy()
                image_numpy = (np.transpose(image_numpy, (1, 2, 0)) if image_numpy.shape[0] == 3 else
                               image_numpy)
                im = Image.fromarray(np.uint8(image_numpy))
            else:
                im = image_list[row][col]
            target.paste(im, ((width_size + padding) * col+((width_size + padding) * col_num - padding+padding2)*row, 0))
    SAVE_QUALITY = 95  # 1（最差）到95（最佳）。 默认值为75，使用中应尽量避免高于95的值
    # PIL.ImageOps.invert(target).save(save_path, quality=SAVE_QUALITY)  # 成品图保存
    target.save(save_path, quality=SAVE_QUALITY)  # 成品图保存


def save_class_activation_images(org_img, activation_map, save_path='../test.png'):
    # Grayscale activation map
    heatmap, heatmap_on_image = apply_colormap_on_image(org_img, activation_map, 'hsv')
    image_list = [org_img, heatmap, heatmap_on_image]
    # image_list=[org_img, heatmap_on_image]
    save_image_all(image_list, height_size=256, width_size=128, save_path=save_path)


def apply_colormap_on_image(org_im, activation, colormap_name):
    """
        Apply heatmap on image
    Args:
        org_img (PIL img): Original image
        activation_map (numpy arr): Activation map (grayscale) 0-255
        colormap_name (str): Name of the colormap
    """
    # Get colormap
    color_map = mpl_color_map.get_cmap(colormap_name)
    no_trans_heatmap = color_map(activation)
    # Change alpha channel in colormap to make sure original image is displayed
    heatmap = copy.copy(no_trans_heatmap)
    heatmap[:, :, 3] = 0.4
    heatmap = Image.fromarray((heatmap*255).astype(np.uint8))
    no_trans_heatmap = Image.fromarray((no_trans_heatmap*255).astype(np.uint8))

    # Apply heatmap on image
    # heatmap_on_image = Image.new("RGBA", org_im.size)
    heatmap_on_image = Image.new("RGBA", (128,256))
    heatmap_on_image = Image.alpha_composite(heatmap_on_image, org_im.convert('RGBA'))
    heatmap_on_image = Image.alpha_composite(heatmap_on_image, heatmap)
    return no_trans_heatmap, heatmap_on_image


def apply_heatmap(R, sx, sy):
    """
        Heatmap code stolen from https://git.tu-berlin.de/gmontavon/lrp-tutorial

        This is (so far) only used for LRP
    """
    b = 10*((np.abs(R)**3.0).mean()**(1.0/3))
    my_cmap = plt.cm.seismic(np.arange(plt.cm.seismic.N))
    my_cmap[:, 0:3] *= 0.85
    my_cmap = ListedColormap(my_cmap)
    plt.figure(figsize=(sx, sy))
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    plt.axis('off')
    heatmap = plt.imshow(R, cmap=my_cmap, vmin=-b, vmax=b, interpolation='nearest')
    return heatmap
    # plt.show()


def format_np_output(np_arr):
    """
        This is a (kind of) bandaid fix to streamline saving procedure.
        It converts all the outputs to the same format which is 3xWxH
        with using sucecssive if clauses.
    Args:
        im_as_arr (Numpy array): Matrix of shape 1xWxH or WxH or 3xWxH
    """
    # Phase/Case 1: The np arr only has 2 dimensions
    # Result: Add a dimension at the beginning
    if len(np_arr.shape) == 2:
        np_arr = np.expand_dims(np_arr, axis=0)
    # Phase/Case 2: Np arr has only 1 channel (assuming first dim is channel)
    # Result: Repeat first channel and convert 1xWxH to 3xWxH
    if np_arr.shape[0] == 1:
        np_arr = np.repeat(np_arr, 3, axis=0)
    # Phase/Case 3: Np arr is of shape 3xWxH
    # Result: Convert it to WxHx3 in order to make it saveable by PIL
    if np_arr.shape[0] == 3:
        np_arr = np_arr.transpose(1, 2, 0)
    # Phase/Case 4: NP arr is normalized between 0-1
    # Result: Multiply with 255 and change type to make it saveable by PIL
    if np.max(np_arr) <= 1:
        np_arr = (np_arr*255).astype(np.uint8)
    return np_arr


def save_image(im, path):
    """
        Saves a numpy matrix or PIL image as an image
    Args:
        im_as_arr (Numpy array): Matrix of shape DxWxH
        path (str): Path to the image
    """
    if isinstance(im, (np.ndarray, np.generic)):
        im = format_np_output(im)
        im = Image.fromarray(im)
    im.save(path)


def preprocess_image(pil_im, resize_im=True):
    """
        Processes image for CNNs

    Args:
        PIL_img (PIL_img): PIL Image or numpy array to process
        resize_im (bool): Resize to 224 or not
    returns:
        im_as_var (torch variable): Variable that contains processed float tensor
    """
    # Mean and std list for channels (Imagenet)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    # Ensure or transform incoming image to PIL image
    if type(pil_im) != Image.Image:
        try:
            pil_im = Image.fromarray(pil_im)
        except Exception as e:
            print("could not transform PIL_img to a PIL Image object. Please check input.")

    # Resize image
    if resize_im:
        # pil_im = pil_im.resize((224, 224), Image.ANTIALIAS)
        pil_im = pil_im.resize((128, 256), Image.ANTIALIAS)

    im_as_arr = np.float32(pil_im)
    im_as_arr = im_as_arr.transpose(2, 0, 1)  # Convert array to D,W,H
    # Normalize the channels
    for channel, _ in enumerate(im_as_arr):
        im_as_arr[channel] /= 255
        im_as_arr[channel] -= mean[channel]
        im_as_arr[channel] /= std[channel]
    # Convert to float tensor
    im_as_ten = torch.from_numpy(im_as_arr).float()
    # Add one more channel to the beginning. Tensor shape = 1,3,224,224
    im_as_ten.unsqueeze_(0)
    # Convert to Pytorch variable
    im_as_var = Variable(im_as_ten, requires_grad=True)
    return im_as_var


def recreate_image(im_as_var):
    """
        Recreates images from a torch variable, sort of reverse preprocessing
    Args:
        im_as_var (torch variable): Image to recreate
    returns:
        recreated_im (numpy arr): Recreated image in array
    """
    reverse_mean = [-0.485, -0.456, -0.406]
    reverse_std = [1/0.229, 1/0.224, 1/0.225]
    recreated_im = copy.copy(im_as_var.data.numpy()[0])
    for c in range(3):
        recreated_im[c] /= reverse_std[c]
        recreated_im[c] -= reverse_mean[c]
    recreated_im[recreated_im > 1] = 1
    recreated_im[recreated_im < 0] = 0
    recreated_im = np.round(recreated_im * 255)

    recreated_im = np.uint8(recreated_im).transpose(1, 2, 0)
    return recreated_im


def get_positive_negative_saliency(gradient):
    """
        Generates positive and negative saliency maps based on the gradient
    Args:
        gradient (numpy arr): Gradient of the operation to visualize

    returns:
        pos_saliency ( )
    """
    pos_saliency = (np.maximum(0, gradient) / gradient.max())
    neg_saliency = (np.maximum(0, -gradient) / -gradient.min())
    return pos_saliency, neg_saliency


