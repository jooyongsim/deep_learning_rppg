"""
PyTorch Dataset classes for dataloader
"""
import torch
import torchvision
from torch.utils.data import Dataset
from torchvision.transforms import RandomRotation, ToPILImage, ToTensor, ColorJitter
import cv2
import numpy as np
import os
import json
from skimage import data, io, filters
from src.fct import get_data, interpolation_ppg

class DatasetPhysNetED(Dataset):
    """
        Dataset class for PhysNet neural network.
    """
    def __init__(self, cfgdict, start = 0, end = None,
                overlap= 0.5, ppg_offset = 6,
                hflip=False, rand_shift=False):

        self.hflip = hflip
        self.random_shift = rand_shift
        # Load video image list
        vdir = cfgdict['videodataDIR']
        self.vdir = vdir
        vfl = sorted(os.listdir(vdir))
                
        if end == None:
            end = len(vfl)
        self.vfl = vfl[start:end-ppg_offset]
        
        self.start = start
        self.end = end
        # Load PPG signals
        self.sigdir = cfgdict['signalpath']
        with open(self.sigdir, "r") as st_json:
            data_json = json.load(st_json)
        time_ns = list()
        ppg = list()
        for dat in data_json['/FullPackage']:
            time_ns.append(dat['Timestamp'])
            ppg.append(dat['Value']['waveform'])
        self.time_ns = np.array(time_ns)
        self.ppg = np.array(ppg)

        imgt = list()
        for fn in vfl:
            imgt.append(int(os.path.splitext(fn)[0][5:]))
        imgt = np.array(imgt)        
        imgt2 = (imgt - imgt[0])/1e6
        time_ns2 = (time_ns - imgt[0])/1e6

        ppg = interpolation_ppg(imgt2, time_ns2, ppg, normalize = True)
        ppg = np.array(ppg)[self.start + ppg_offset:self.end]
        self.ppg = ppg

        # Image config
        self.depth = int(cfgdict['depth'])
        self.height = int(cfgdict['height'])
        self.width = int(cfgdict['width'])
        self.channel = 3
        self.overlap = overlap
        self.shift = int(self.depth*(1-overlap))  # overlap, s.t., 0=< overlap < 1
        self.num_samples = (end - start - self.depth)//self.shift+1

        self.crop = bool(cfgdict['crop'])
#         if self.crop:        
        # Crop Face Rectangle - from 1st image
        vfpath = os.path.join(vdir,vfl[0])
        img = io.imread(vfpath)

        dpath = "./config/haarcascade_frontalface_alt.xml"
        if not os.path.exists(dpath):
            print("Cascade file not present!")
        face_cascade = cv2.CascadeClassifier(dpath)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        detected = list(face_cascade.detectMultiScale(gray, 1.1, 4))

        if len(detected) > 0:
            detected.sort(key=lambda a: a[-1] * a[-2])
            face_rect = detected[-1]
        self.face_rect = face_rect
#         x, y, w, h = face_rect

    def __len__(self):
        return self.num_samples # min(int(ppg.shape[0]/2), len(fl)-1)
    
    def __getitem__(self, idx):
        # -------------------------------
        # Fill video with frames
        # -------------------------------
        # conv3d input: N x C x D x H X W
        x, y, w, h = self.face_rect
        
        shift = self.shift
        depth = self.depth
        height = self.height
        width = self.width
        channel = self.channel
        # vfl = self.vfl
        # ppg = self.ppg
        vdir = self.vdir
            
        # TODO: Add temporal jitter
        video = torch.empty(channel, depth, height, width, dtype=torch.float)
        
        if self.random_shift:
            rand_offset = int(depth*(1-self.overlap)*0.5)
            rand_shift = random.randint(-rand_offset, rand_offset)
        else:
            rand_shift = 0
        
        start_frame = idx * shift + rand_shift
        end_frame = idx * shift + depth + rand_shift

        while self.start+start_frame < self.start:
            start_frame += 1
            end_frame += 1

        while self.start + end_frame >= self.end or self.start + end_frame >= len(self.ppg) + self.start:
            start_frame -= 1
            end_frame -= 1

        for cnt, fn in enumerate(self.vfl[start_frame : end_frame]):
            vfpath = os.path.join(vdir,fn)
            img = io.imread(vfpath)
            if self.crop:        
                img = img[y:y + h, x: x + w, :]
            img = cv2.resize(img, (height, width), interpolation=cv2.INTER_CUBIC)
            img = ToTensor()(img)
            img = torch.sub(img, torch.mean(img, (1, 2)).view(3, 1, 1))  # spatial intensity norm for each channel

            # TODO: add jitter and flip
            if self.hflip:
                rand_flip = bool(random.getrandbits(1))
                if rand_flip:
                    img = torchvision.transforms.functional.hflip(img)

            video[:, cnt, :, :] = img  # video -> C, D, H, W
                         
            # Swap axes because  numpy image: H x W x C | torch image: C X H X W
#             img = torch.from_numpy(img.astype(np.float32))
#             img = img.permute(2, 0, 1)
#             img =/ 255. # convert image to [0, 1]
#                     img = tr.sub(img, tr.mean(img, (1, 2)).view(3, 1, 1))  # Color channel centralization
        target = self.ppg[start_frame :  end_frame]
        target = torch.tensor(target, dtype=torch.float)
        return video, target