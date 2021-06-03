import torch
from torch.nn.parameter import Parameter
import numpy as np
import os
import cv2
from collections import OrderedDict
from time import time

from LapNet import LAPNet
from create_dataset import createDataset

class LapNet_Test:
    def __init__(self, model_name):
        # torch.cuda.set_device(args.gpu_idx)
        torch.cuda.set_device(0)

        # self.INPUT_CHANNELS = 3
        # self.OUTPUT_CHANNELS = 2
        # self.LEARNING_RATE = args.lr #1e-5
        # self.BATCH_SIZE = args.batch_size #20
        # self.NUM_EPOCHS = args.epoch #100
        # self.LOG_INTERVAL = 20
        # self.INS_CH = 32
        # self.SIZE = [args.img_size[0], args.img_size[1]] #[224, 224]
        # self.NUM_WORKERS = args.num_workers #20

        self.INPUT_CHANNELS = 3
        self.OUTPUT_CHANNELS = 2
        self.LEARNING_RATE = 3e-4
        self.BATCH_SIZE = 32
        self.NUM_EPOCHS = 10000000000000
        self.LOG_INTERVAL = 20
        self.INS_CH = 32
        self.SIZE = [1024,512]
        self.NUM_WORKERS = 32

        self.model_name = model_name

        self.root_path = '../'

        self.model = LAPNet(input_ch=self.INPUT_CHANNELS, output_ch=self.OUTPUT_CHANNELS,internal_ch = 8).cuda()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.LEARNING_RATE, betas=(0.9, 0.99), amsgrad=True)

        chkpt_filename = self.root_path + 'model/' + self.model_name

        if not os.path.exists(self.root_path + 'model/'):
            os.mkdir(self.root_path + 'model/')
        if os.path.isfile(chkpt_filename):
            checkpoint = torch.load(chkpt_filename)
            self.start_epoch = checkpoint['epoch']
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.model.load_state_dict(checkpoint['net'])
            self.load_state_dict(self.model, self.state_dict(self.model))

    def state_dict(self, model, destination=None, prefix='', keep_vars=False):
        own_state = model.module if isinstance(model, torch.nn.DataParallel) \
            else model
        if destination is None:
            destination = OrderedDict()
        for name, param in own_state._parameters.items():
            if param is not None:
                destination[prefix + name] = param if keep_vars else param.data
        for name, buf in own_state._buffers.items():
            if buf is not None:
                destination[prefix + name] = buf
        for name, module in own_state._modules.items():
            if module is not None:
                self.state_dict(module, destination, prefix + name + '.', keep_vars=keep_vars)
        return destination

    def load_state_dict(self, model, state_dict, strict=True):
        own_state = model.module.state_dict() if isinstance(model, torch.nn.DataParallel) \
            else model.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, Parameter):
                    # backwards compatibility for serialized parameters
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    raise RuntimeError('While copying the parameter named {}, '
                                        'whose dimensions in the model are {} and '
                                        'whose dimensions in the checkpoint are {}.'
                                        .format(name, own_state[name].size(), param.size()))
            elif strict:
                raise KeyError('unexpected key "{}" in state_dict'
                                .format(name))
        if strict:
            missing = set(own_state.keys()) - set(state_dict.keys())
            if len(missing) > 0:
                raise KeyError('missing keys in state_dict: "{}"'.format(missing))
    
    def predict(self, image):
        train_dataset = createDataset("", size=lapnet_test.SIZE, image=image)
        train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=24, pin_memory=True,
                                                    shuffle=False, num_workers=0)
                
        img = list(enumerate(train_dataloader))[0][1]

        img_tensor = torch.tensor(img).cuda()

        sem_pred = lapnet_test.model(img_tensor)

        seg_map = torch.squeeze(sem_pred, 0).cpu().detach().numpy()

        seg_show = seg_map[1]

        _, seg_show2 = cv2.threshold(seg_show + 1, 0, 0, cv2.THRESH_TOZERO)
        seg_show2 = cv2.normalize(seg_show2, seg_show2, 0, 1, cv2.NORM_MINMAX)
        seg_show2 = cv2.convertScaleAbs(seg_show2, seg_show2, 255)
        result_img = cv2.applyColorMap(seg_show2, cv2.COLORMAP_MAGMA)

        return result_img

    def predict_with_path(self, image_path):
        image = cv2.imread(image_path)
        return self.predict(image)
    
    def predict_video(self, video_path, show_video, save_video):
        cap = cv2.VideoCapture(video_path)
        
        video_appendix = video_path.split(".")[-1]
        
        video_base_name = video_path.split("." + video_appendix)[0]
    
        if save_video:
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            out = cv2.VideoWriter(video_base_name + "_output." + video_appendix, fourcc, cap.get(5), (1024, 1024))
        
        frame_num = int(cap.get(7))
        
        solved_num = 0
        
        start_time = time()
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            source = cv2.resize(frame, (1024, 512))
            edge = lapnet_test.predict(source)
            merge = np.vstack([source, edge])
            if save_video:
                out.write(merge)
            solved_num += 1
            if show_video:
                cv2.imshow("merge", merge)
                k = cv2.waitKey(20)
                if k & 0xff == ord("q"):
                    break
            fps = 1.0 * solved_num / (time() - start_time)
            time_needed = int(1.0 * (frame_num - solved_num) / fps)
            minite_needed = int(time_needed / 60)
            second_needed = time_needed % 60
            print("\rProcess :", solved_num, "/", frame_num, "\tFPS =", int(fps), "\tTime needed :", minite_needed, "min", second_needed, "s   ", end="")
        print()
        
        cap.release()
        if save_video:
            out.release()
        cv2.destroyAllWindows()
        
    def save_full_model(self, save_name):
        example = torch.rand(1, 3, self.SIZE[1], self.SIZE[0]).cuda()
        traced_script_moodule = torch.jit.trace(self.model, example)
        traced_script_moodule.save(self.root_path + 'model/' + save_name)
        

if __name__ == "__main__":
    model_name = "LapNet_chkpt_better_epoch6767_GPU0_HED_detect.pth"
    video_path = "../data/test/NVR_ch2_main_20201111164000_20201111170000.avi"
    save_name = "LapNet_Edge_Detect.pt"
    show_video = True
    save_video = False
    save_full_model = False
    
    lapnet_test = LapNet_Test(model_name)
    lapnet_test.model.eval()
    
    if save_full_model:
        lapnet_test.save_full_model(save_name)
    else:
        lapnet_test.predict_video(video_path, show_video, save_video)
