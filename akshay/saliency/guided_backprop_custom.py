import torch
from torch.nn import ReLU

from misc_functions import (get_example_params,
                            convert_to_grayscale,
                            save_gradient_images,
                            get_positive_negative_saliency)

import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import datetime

import os, sys, time 
import numpy as np 
import matplotlib.pyplot as plt 

class GuidedBackprop():
    """
       Produces gradients generated with guided back propagation from the given image
    """
    def __init__(self, model):
        self.model = model
        self.gradients = None
        self.forward_relu_outputs = []
        # Put model in evaluation mode
        self.model.eval()
        self.update_relus()
        self.hook_layers()

    def hook_layers(self):
        def hook_function(module, grad_in, grad_out):
            self.gradients = grad_in[0]
        # Register hook to the first layer
        first_layer = list(self.model.features._modules.items())[0][1][0]
        first_layer.register_backward_hook(hook_function)

    def update_relus(self):
        """
            Updates relu activation functions so that
                1- stores output in forward pass
                2- imputes zero for gradient values that are less than zero
        """
        def relu_backward_hook_function(module, grad_in, grad_out):
            """
            If there is a negative gradient, change it to zero
            """
            # Get last forward output
            corresponding_forward_output = self.forward_relu_outputs[-1]
            corresponding_forward_output[corresponding_forward_output > 0] = 1
            modified_grad_out = corresponding_forward_output * torch.clamp(grad_in[0], min=0.0)
            del self.forward_relu_outputs[-1]  # Remove last forward output
            return (modified_grad_out,)

        def relu_forward_hook_function(module, ten_in, ten_out):
            """
            Store results of forward pass
            """
            self.forward_relu_outputs.append(ten_out)

        # Loop through layers, hook up ReLUs
        for pos, module in self.model.features._modules.items():
            if isinstance(module, ReLU):
                module.register_backward_hook(relu_backward_hook_function)
                module.register_forward_hook(relu_forward_hook_function)

    def generate_gradients(self, input_image, target_class):
        # Forward pass
        model_output = self.model(input_image)
        # Zero gradients
        self.model.zero_grad()
        # Target for backprop
        one_hot_output = torch.FloatTensor(1, model_output.size()[-1]).zero_()
        one_hot_output[0][target_class] = 1
        # Backward pass
        model_output.backward(gradient=one_hot_output)
        # Convert Pytorch variable to numpy array
        # [0] to get rid of the first channel (1,3,224,224)
        gradients_as_arr = self.gradients.data.numpy()[0]
        return gradients_as_arr


if __name__ == '__main__':


    # custom for Hai's model
    input_size = 224
    num_classes = 102

    # Load data
    dataPath = '../../caltech_divide/Caltech101'
    val_dataset = datasets.ImageFolder(
        os.path.join(dataPath, 'test'),
        transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]))

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=1,shuffle=True,
        num_workers=1, pin_memory=False, sampler=None)

    # vgg model
    model = models.vgg16()
    n_inputs = model.classifier[6].in_features
    fea_idx = 3
    model.classifier = nn.Sequential(
        nn.Linear(512 * 7 * 7, n_inputs),
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(n_inputs, n_inputs),
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(n_inputs, num_classes),
    )
    
    model.features = torch.nn.DataParallel(model.features)
    # model.cuda()

    model_location = "../../saved_models/transfer_2019_04_11-2116/model_best.pth.tar"
    if os.path.isfile(model_location):
        print("=> loading checkpoint '{}'".format(model_location))
        checkpoint = torch.load(model_location, map_location="cpu")       
        best_prec1 = checkpoint['best_prec1']
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {})"
                .format(model_location, checkpoint['epoch']))
        del checkpoint
    else:
        print("=> no checkpoint found at '{}'".format(args.model))

    # Vanilla backprop
    GBP = GuidedBackprop(model)
    # VBP = VanillaBackprop(model)

    # get an image 
    image_index = 0
    for i, (input, target) in enumerate(val_loader):

        image_index += 1
        print("WORKING ON IMAGE {}".format(image_index))
        input.requires_grad = True
        prep_img = input
        target_class = target.item() 
        
        # Generate gradients
        # vanilla_grads = VBP.generate_gradients(prep_img, target_class)
        guided_grads = GBP.generate_gradients(prep_img, target_class)
        
        # Convert to grayscale
        grayscale_guided_grads = convert_to_grayscale(guided_grads)
        # Save grayscale gradients
        save_gradient_images(grayscale_guided_grads, "GRAD_{}".format(image_index))

        # Unnormalize input image 
        input_copy = np.copy(input.detach().numpy()) 
        input_copy = input_copy[0] 
        input_copy[0] = input_copy[0] * 0.229 
        input_copy[1] = input_copy[1] * 0.224 
        input_copy[2] = input_copy[2] * 0.225 
        input_copy[0] = input_copy[0] + 0.485 
        input_copy[1] = input_copy[1] + 0.456 
        input_copy[2] = input_copy[2] + 0.406 
        input_copy = np.rollaxis(input_copy, axis=2)
        input_copy = np.rollaxis(input_copy, axis=2)
        plt.imsave("./sample_images/IMAGE_{}.png".format(image_index), input_copy)
    








    # target_example = 0  # Snake
    # (original_image, prep_img, target_class, file_name_to_export, pretrained_model) =\
    #     get_example_params(target_example)

    # # Guided backprop
    # GBP = GuidedBackprop(pretrained_model)
    # # Get gradients
    # guided_grads = GBP.generate_gradients(prep_img, target_class)
    # # Save colored gradients
    # save_gradient_images(guided_grads, file_name_to_export + '_Guided_BP_color')
    # # Convert to grayscale
    # grayscale_guided_grads = convert_to_grayscale(guided_grads)
    # # Save grayscale gradients
    # save_gradient_images(grayscale_guided_grads, file_name_to_export + '_Guided_BP_gray')
    # # Positive and negative saliency maps
    # pos_sal, neg_sal = get_positive_negative_saliency(guided_grads)
    # save_gradient_images(pos_sal, file_name_to_export + '_pos_sal')
    # save_gradient_images(neg_sal, file_name_to_export + '_neg_sal')
    # print('Guided backprop completed')
