import torch, cv2, os, sys, numpy as np, matplotlib.pyplot as plt
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
from ModelZoo.utils import load_as_tensor, Tensor2PIL, PIL2Tensor, _add_batch_one
from ModelZoo import get_model, load_model, print_network
from SaliencyModel.utils import vis_saliency, vis_saliency_kde, click_select_position, grad_abs_norm, grad_norm, prepare_images, make_pil_grid, blend_input
from SaliencyModel.utils import cv2_to_pil, pil_to_cv2, gini
from SaliencyModel.attributes import attr_grad
from SaliencyModel.BackProp import I_gradient, attribution_objective, Path_gradient
from SaliencyModel.BackProp import saliency_map_PG as saliency_map
from SaliencyModel.BackProp import GaussianBlurPath
from SaliencyModel.utils import grad_norm, IG_baseline, interpolation, isotropic_gaussian_kernel
model = load_model('ELAN@Base')
window_size = 16  # Define windoes_size of D
img_lr, img_hr = prepare_images('/home/dlg/SISR/LAM_Demo/test_images/8.png')  # Change this image name
tensor_lr = PIL2Tensor(img_lr)[:3] ; tensor_hr = PIL2Tensor(img_hr)[:3]
cv2_lr = np.moveaxis(tensor_lr.numpy(), 0, 2) ; cv2_hr = np.moveaxis(tensor_hr.numpy(), 0, 2)
plt.imshow(cv2_hr)
w = 110  # The x coordinate of your select patch, 125 as an example
h = 150  # The y coordinate of your select patch, 160 as an example
         # And check the red box
         # Is your selected patch this one? If not, adjust the `w` and `h`


draw_img = pil_to_cv2(img_hr)
cv2.rectangle(draw_img, (w, h), (w + window_size, h + window_size), (0, 0, 255), 2)
position_pil = cv2_to_pil(draw_img)
#position_pil.show()
sigma = 1.2 ; fold = 50 ; l = 9 ; alpha = 0.5


attr_objective = attribution_objective(attr_grad, h, w, window=window_size)
gaus_blur_path_func = GaussianBlurPath(sigma, fold, l)
interpolated_grad_numpy, result_numpy, interpolated_numpy = Path_gradient(((tensor_lr.numpy())*255).astype(np.uint8), model, attr_objective, gaus_blur_path_func, cuda=False)
grad_numpy, result = saliency_map(interpolated_grad_numpy, result_numpy)
image_np = (result.numpy() * 255).astype(np.uint8)

# 如果张量为4维，通常是(batch_size, channels, height, width)的形式
# 由于这里的张量是(batch_size=1, channels=3, height, width)，我们选择索引0取出图像
#image_np = image_np[0]
#print(tensor_lr)
#print(tensor_lr.numpy())
#print(image_np)
#需要调试时，可以选择将张量打印
abs_normed_grad_numpy = grad_abs_norm(grad_numpy)
saliency_image_abs = vis_saliency(abs_normed_grad_numpy, zoomin=4)
saliency_image_kde = vis_saliency_kde(abs_normed_grad_numpy)
blend_abs_and_input = cv2_to_pil(pil_to_cv2(saliency_image_abs) * (1.0 - alpha) + pil_to_cv2(img_lr.resize(img_hr.size)) * alpha)
blend_kde_and_input = cv2_to_pil(pil_to_cv2(saliency_image_kde) * (1.0 - alpha) + pil_to_cv2(img_lr.resize(img_hr.size)) * alpha)

#print(result)
image_array = result.squeeze().numpy()

# 对数组进行四舍五入，使其范围限制在0到255之间
rounded_array = np.round(image_array).astype(np.uint8)

# 将颜色通道顺序调整为RGB
rounded_array_rgb = np.moveaxis(rounded_array, 0, -1)

# 显示图片
plt.imshow(rounded_array_rgb)
plt.axis('off')
plt.show()

pil = make_pil_grid(
    [position_pil,
     saliency_image_abs,
     blend_abs_and_input,
     blend_kde_and_input,
     Tensor2PIL(torch.clamp(result, min=0., max=1.))]
)
pil.show()
gini_index = gini(abs_normed_grad_numpy)
diffusion_index = (1 - gini_index) * 100
print(f"The DI of this case is {diffusion_index}")