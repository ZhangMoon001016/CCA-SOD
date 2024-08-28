import torch
import cv2
import time
import os
import os.path as osp
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
from argparse import ArgumentParser
from collections import OrderedDict
from models import model as net
from tqdm import tqdm
import glob

@torch.no_grad()
def test(args, model, image_list,dataset):
    mean = [0.406, 0.456, 0.485]
    std = [0.225, 0.224, 0.229]
    cnt=0
    cost_time = list()
    for idx in tqdm(range(len(image_list))):
        image = cv2.imread(image_list[idx])
        start_time = time.perf_counter()
        # resize and normalize the image
        img = cv2.resize(image, (args.width, args.height))
        img = img.astype(np.float32) / 255.
        img -= mean
        img /= std

        img = img[:, :, ::-1].copy()
        img = img.transpose((2, 0, 1))
        img = torch.from_numpy(img).unsqueeze(0)
        img = Variable(img)

        if args.gpu:
            img = img.cuda()

        img_out = model(img)[:, 0, :, :].unsqueeze(dim=0)

        img_out = F.interpolate(img_out, size=image.shape[:2], mode='bilinear', align_corners=False)

        sal_map = (img_out * 255).data.cpu().numpy()[0, 0].astype(np.uint8)
        cost_time.append(time.perf_counter() - start_time)
        # cv2.imwrite(image_list[idx].replace(".jpg", "_edn.png"), sal_map)
        #print(image_list)
        #path = image_list[idx].replace(dataset, "EDN")

        cnt=cnt+1
        print(cnt,'11111111111')
        path=image_list[idx].replace("RGB", "EDN")
        cv2.imwrite(path.replace(".jpg", ".png"), sal_map)
        print(path)
    cost_time.pop(0)
    print('Mean running time is: ', cost_time)
    print('Mean running time is: ', np.mean(cost_time))

def main(args, example_path="E:/2/data/"):
    # read all the images in the example folder
    #datasets=['0.999','0.95','0.90','0.85','0.80','0.75','0.70','0.65','0.60','0.55','0.50','0.45','0.40']
    datasets = ['NLPR']
    for dataset in datasets:
        example_dir=example_path+dataset+'/'+'RGB/'
        print(example_dir)
        save_path = 'F:/zmy/大修/'+dataset
        print(save_path)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        image_list = glob.glob("{}*.jpg".format(example_dir))

        model = net.EDN(arch=args.arch)

        if not osp.isfile(args.pretrained):
            print('Pre-trained model file does not exist...')
            print("start to download the pretrained model!")
            os.system("wget -c https:/github.com/yuhuan-wu/EDN/releases/download/v1.0/EDN-VGG16.pth -O pretrained/EDN-VGG16.pth")

        state_dict = torch.load(args.pretrained)
        new_keys = []
        new_values = []
        for key, value in zip(state_dict.keys(), state_dict.values()):
            new_keys.append(key.replace('module.', ''))
            new_values.append(value)
        new_dict = OrderedDict(list(zip(new_keys, new_values)))
        model.load_state_dict(new_dict)

        if args.gpu:
            model = model.cuda()

        model.eval()
        test(args, model, image_list,dataset)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--arch', default='vgg16', help='the backbone name of EDN, vgg16, resnet50, or mobilenetv2')
    parser.add_argument('--example_dir', default="E:/2/data/ReDWeb-S/testset/RGB_snr/", help='Data directory')
    parser.add_argument('--width', type=int, default=384, help='Width of RGB image')
    parser.add_argument('--height', type=int, default=384, help='Height of RGB image')
    parser.add_argument('--gpu', default=True, type=lambda x: (str(x).lower() == 'true'),
                        help='Run on CPU or GPU. If TRUE, then GPU')
    parser.add_argument('--pretrained', default="pretrained/EDN-VGG16.pth", help='Pretrained model')

    args = parser.parse_args()
    print('Called with args:')
    print(args)

    main(args)
