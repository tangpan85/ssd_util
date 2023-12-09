from pathlib import Path

from mmdet.apis import init_detector, inference_detector

config_file = r'config/deformable-detr-refine-twostage_r50_8xb4_sample1e-3_v3det_50e.py'
checkpoint_file = 'model/deformable_detr_swin_425.pth'
model = init_detector(config_file, checkpoint_file, device='cpu')  # or device='cuda:0'


r = inference_detector(model,r'D:\Python\dates\test\images\test\6901285991219_camera1-3.jpg')

print("ok")
print(r)