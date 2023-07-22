"""
Created on Thu Oct 26 11:06:51 2017

@author: Utku Ozbulak - github.com/utkuozbulak
"""
from PIL import Image
import numpy as np
import torch
from IPython import embed
from .misc_functions import preprocess_image, apply_colormap_on_image, save_image_all, save_image_single_row


def save_heatmap(records):
    '''
    :param records: 一个字典（图片-model列表（列表中每个元素存放 模型、待提取层、模型类型））
    :return:
    '''
    image_lists=[]
    for img_path in records:
        print('-->'+img_path)
        image_list=[]
        original_image = Image.open(img_path).convert('RGB').resize((128, 256), Image.ANTIALIAS)
        prep_img = preprocess_image(original_image)
        image_list.append(original_image)
        for rec in records[img_path]:
            model, target_layer, model_type=rec
            cam_model = GradCam(model, target_layer, model_type)  # Grad cam
            cam_mask = cam_model.generate_cam(prep_img)  # Generate cam mask
            heatmap, heatmap_on_image = apply_colormap_on_image(original_image, cam_mask, 'hsv')
            image_list.append(heatmap_on_image)
        image_lists.append(image_list)
    # save_image_all(image_lists, save_path='/home/wuqi/kr/kr1/CA/img/test.png')
    save_image_single_row(image_lists, save_path='/home/wuqi/kr/kr1/CA/img/test2.png')
    print('Save all heatmap!')


def grad_cam(original_image, model, target_layer, model_type):
    # 0068_c6s1_012476_01
    # img_path[img_path.rfind('/') + 1:img_path.rfind('.')]
    # img_name="0068_c6s1_012476_01"
    # img_path = "/home/wuqi/kr/dataset/market/bounding_box_train/"+img_name+".jpg"

    # original_image = Image.open(img_path).convert('RGB')
    prep_img = preprocess_image(original_image)

    cam_model = GradCam(model, target_layer, model_type) # Grad cam
    cam_mask = cam_model.generate_cam(prep_img) # Generate cam mask
    # Save mask
    original_image=original_image.resize((128, 256), Image.ANTIALIAS)
    # save_class_activation_images(original_image, cam)
    # print('Grad cam completed')
    return cam_mask


class CamExtractor():
    """
        Extracts cam features from the model
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None

    def save_gradient(self, grad):
        self.gradients = grad

    def forward_pass_on_convolutions(self, x):
        """
            Does a forward pass on convolutions, hooks the function at given layer
        """
        conv_output = None
        for module_name, module in self.model.base._modules.items():
            # print(module_name)
            # if 'layer' in module_name: continue
            x = module(x)  # Forward
            if module_name == self.target_layer:
                x.register_hook(self.save_gradient)
                conv_output = x  # Save the convolution output on that layer
        # embed()
        return conv_output, x

    def forward_pass(self, x):
        """
            Does a full forward pass on the model
        """
        # Forward pass on the convolutions
        conv_output, x = self.forward_pass_on_convolutions(x)
        # x = x.view(x.size(0), -1)  # Flatten

        # Forward pass on the classifier
        global_feat = self.model.global_pool(x)  # (b, 2048, 1, 1)
        global_feat = global_feat.view(global_feat.shape[0], -1)  # flatten to (bs, 2048)
        x = self.model.bottleneck(global_feat)
        x = self.model.classifier(x)
        return conv_output, x


class CamExtractor_ca():
    """
        Extracts cam features from the model
    """

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None

    def save_gradient(self, grad):
        self.gradients = grad

    def forward_pass_on_convolutions_ca(self, x):
        """
            Does a forward pass on convolutions, hooks the function at given layer
        """
        conv_output = None
        x_id = x
        x_po = x.contiguous()
        for module_name, module in self.model.base._modules.items():
            if 'block' in module_name:
                x_id, x_po = module(x_id, x_po)
                if module_name == self.target_layer:
                    x_id.register_hook(self.save_gradient)
                    conv_output = x_id  # Save the convolution output on that layer
        return conv_output, x_id, x_po

    def forward_pass(self, x):
        """
            Does a full forward pass on the model
        """
        # Forward pass on the convolutions
        conv_output, x_id, x_po = self.forward_pass_on_convolutions_ca(x)
        # x = x.view(x.size(0), -1)  # Flatten

        # Forward pass on the classifier
        global_feat_id = self.model.global_pool(x_id)  # (b, 2048, 1, 1)
        global_feat_id = global_feat_id.view(global_feat_id.shape[0], -1)  # flatten to (bs, 2048)
        feat_bn_id = self.model.bottleneck(global_feat_id)
        cls_score_id = self.model.classifier(feat_bn_id)
        return conv_output, cls_score_id

class GradCam():
    """
        Produces class activation map
    """
    def __init__(self, model, target_layer, model_type):
        self.model = model
        self.model.eval()
        # Define extractor
        self.extractor = CamExtractor_ca(self.model, target_layer) if 'ca' in model_type else \
            CamExtractor(self.model, target_layer)

    def generate_cam(self, input_image, target_class=None):
        # Full forward pass
        # conv_output is the output of convolutions at specified layer (batch_size,C,H,W)
        # model_output is the final output of the model (1, 1000)
        conv_output, model_output = self.extractor.forward_pass(input_image)
        if target_class is None:
            target_class = np.argmax(model_output.data.numpy())
        # Target for backprop
        one_hot_output = torch.FloatTensor(1, model_output.size()[-1]).zero_()
        one_hot_output[0][target_class] = 1
        # Zero grads
        self.model.base.zero_grad()
        self.model.bottleneck.zero_grad()
        self.model.classifier.zero_grad()
        # Backward pass with specified target
        model_output.backward(gradient=one_hot_output, retain_graph=True)
        # Get hooked gradients
        guided_gradients = self.extractor.gradients.data.numpy()[0]
        # Get convolution outputs
        target = conv_output.data.numpy()[0]
        # Get weights from gradients
        weights = np.mean(guided_gradients, axis=(1, 2))  # Take averages for each gradient
        # Create empty numpy array for cam
        cam = np.ones(target.shape[1:], dtype=np.float32)
        # Have a look at issue #11 to check why the above is np.ones and not np.zeros
        # Multiply each weight with its conv output and then, sum
        for i, w in enumerate(weights):
            cam += w * target[i, :, :]
        cam = np.maximum(cam, 0)
        cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))  # Normalize between 0-1
        cam = np.uint8(cam * 255)  # Scale between 0-255 to visualize
        # cam = np.uint8(Image.fromarray(cam).resize((input_image.shape[2], input_image.shape[3]), Image.ANTIALIAS))/255
        cam = np.uint8(Image.fromarray(cam).resize((input_image.shape[3], input_image.shape[2]), Image.ANTIALIAS))/255
        # ^ I am extremely unhappy with this line. Originally resizing was done in cv2 which
        # supports resizing numpy matrices with antialiasing, however,
        # when I moved the repository to PIL, this option was out of the window.
        # So, in order to use resizing with ANTIALIAS feature of PIL,
        # I briefly convert matrix to PIL image and then back.
        # If there is a more beautiful way, do not hesitate to send a PR.

        # You can also use the code below instead of the code line above, suggested by @ ptschandl
        # from scipy.ndimage.interpolation import zoom
        # cam = zoom(cam, np.array(input_image[0].shape[1:])/np.array(cam.shape))
        return cam


if __name__ == '__main__':
    grad_cam()
