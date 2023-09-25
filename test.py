import torch
import numpy as np
from models.model import newResNet
from models.softmax_layer import SoftMaxLayer
from transform.zca import ZCA
from utils.utils import l1_norm
from torchvision import transforms
from torchvision.transforms import Resize, ToTensor, Grayscale, ToPILImage
from PIL import Image
import cv2

import warnings
warnings.filterwarnings("ignore")

transform = transforms.Compose([
    Grayscale(num_output_channels=3),
    Resize((224, 224)),
    ToTensor()
])



# Reading Image data for both visible and infrared
vis_input = Image.open('data/VIS19.png')
ir_input = Image.open('data/IR19.png')

transform_vis = transforms.Compose([
    ToPILImage(),
    # Grayscale(num_output_channels=3),
    Resize((vis_input.size[0], vis_input.size[1])),
    ToTensor()
])

transform_ir = transforms.Compose([
    ToPILImage(),
    # Grayscale(num_output_channels=3),
    Resize((ir_input.size[0], ir_input.size[1])),
    ToTensor()
])

#Extract features from layer 3 and layer 4 for both images

#features for visible image
model_visible = newResNet()

layer3 = model_visible.net.layer3.register_forward_hook(model_visible.activation('layer3'))
layer4 = model_visible.net.layer4.register_forward_hook(model_visible.activation('layer4'))

vis = transform(vis_input)
vis = torch.unsqueeze(vis, 0)
print(vis.shape)

vis_ = model_visible(vis)

activation_visible = model_visible.get_activation()

layer3.remove()
layer4.remove()

#features for infrared image
model_infra = newResNet()

layer3_ = model_infra.net.layer3.register_forward_hook(model_infra.activation('layer3'))
layer4_ = model_infra.net.layer4.register_forward_hook(model_infra.activation('layer4'))

ir = transform(ir_input)
ir = torch.unsqueeze(ir, 0)

print(ir.shape)

ir_ = model_infra(ir)

activation_infra = model_infra.get_activation()

layer3_.remove()
layer4_.remove()


# print(activation_visible)
# print(activation_infra)

# Applying ZCA + l1_norm on extracted features

zca_ = ZCA()

#Visible image
vis_layer3 = activation_visible['layer3'].squeeze().permute(2, 1, 0)
vis_layer4 = activation_visible['layer4'].squeeze().permute(2, 1, 0)

trf = zca_.fit(vis_layer3)
vis_layer3_zca = trf.transform(vis_layer3)
vis_layer3_l1 = l1_norm(vis_layer3_zca)

trf = zca_.fit(vis_layer4)
vis_layer4_zca = trf.transform(vis_layer4)
vis_layer4_l1 = l1_norm(vis_layer4_zca)

# X_reconstructed = trf.inverse_transform(X_whitened)

#Infrared image
ir_layer3 = activation_infra['layer3'].squeeze().permute(2, 1, 0)
ir_layer4 = activation_infra['layer4'].squeeze().permute(2, 1, 0)

trf = zca_.fit(ir_layer3)
ir_layer3_zca = trf.transform(ir_layer3)
ir_layer3_l1 = l1_norm(ir_layer3_zca)


trf = zca_.fit(ir_layer4)
ir_layer4_zca = trf.transform(ir_layer4)
ir_layer4_l1 = l1_norm(ir_layer4_zca)

# print(ir_layer3_l1)

# Resizing features to match the original input images

x = torch.from_numpy(vis_layer3_l1).type(torch.float32)
print(x.shape)
vis_layer3_l11 = transform_vis(torch.from_numpy(vis_layer3_l1).type(torch.float32))
vis_layer4_l11 = transform_vis(torch.from_numpy(vis_layer4_l1).type(torch.float32))

ir_layer3_l11 = transform_ir(torch.from_numpy(ir_layer3_l1).type(torch.float32))
ir_layer4_l11 = transform_ir(torch.from_numpy(ir_layer4_l1).type(torch.float32))


#passing resized images to SoftMax Layer

soft = SoftMaxLayer()


vis_layer3_weighted_map = soft(vis_layer3_l11)
vis_layer4_weighted_map = soft(vis_layer4_l11)

ir_layer3_weighted_map = soft(ir_layer3_l11)
ir_layer4_weighted_map = soft(ir_layer4_l11)

fused_image = np.multiply(vis_layer3_weighted_map.squeeze().permute(1, 0), torch.from_numpy(np.array(vis_input))) + np.multiply(ir_layer3_weighted_map.squeeze().permute(1, 0) , torch.from_numpy(np.array(ir_input)))

im = Image.fromarray(np.array(fused_image))
image = im.convert("RGB")

# Save the image as a PNG.
image.save("image.png")

print(fused_image)



