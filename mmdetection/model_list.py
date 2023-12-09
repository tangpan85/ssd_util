import os
from pathlib import Path

from mmdet.apis import DetInferencer
os.environ['TORCH_HOME'] = r'D:\Python\dates\model'

# model_list = ["rtmdet_l_8xb32-300e_coco",
#               "ssd300_cocodeformable-detr-refine_r50_16xb2-50e_coco"
#               "deformable-detr-refine_r50_16xb2-50e_coco"]
model_list = {"detr":"deformable-detr-refine_r50_16xb2-50e_coco"}

def get_file_model():
    # 从本地文件中获取模型
    model_path = r"/model"
    for file in Path(model_path).glob('*.pth'):
        model_list[file.stem] = str(file.resolve())



def get_model(model):
    inferencer = DetInferencer(model=model)
    return inferencer


def main_inferencer(img_path):
    get_file_model()

    # 遍历字典 model_list
    for model_name, model_path in model_list.items():
        path = rf'rdata\{model_name}'
        path_p = Path(path)
        path_p.mkdir(exist_ok=True, parents=True)
        inferencer = get_model(model_path)
        inferencer(img_path, out_dir=path, no_save_pred=False)

        print(f'{model_name} done')



main_inferencer(r'D:\Python\dates\test\images\test')
