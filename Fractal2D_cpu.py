import sys
sys.path.insert(1, "./build/lib.linux-aarch64-3.9")
import torch
import torchvision
import numpy as np
import PyFractal2DRenderer as fr
import os
from typing import Any, Callable, cast, Dict, List, Optional, Tuple
import math
import random
import re
from typing import List

from PIL import Image
from torchvision import datasets, transforms
from torchvision.datasets.folder import make_dataset, find_classes, is_image_file, has_file_allowed_extension


def worker_init_fdb_fn(worker_id):
    info = torch.utils.data.get_worker_info()
    seed = info.dataset.patchgen_seed + worker_id
    info.dataset.patchgen_rng = np.random.default_rng(seed)

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')


def pil_loader(path: str) -> Image.Image:
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


# TODO: specify the return type
def accimage_loader(path: str) -> Any:
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path: str) -> Any:
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)
    
class Fractal2D_cpu(datasets.DatasetFolder):
    """Dataset of generating 2D fractal image on time
    """
    RENDER_MODE_POINT = 0
    RENDER_MODE_LINE = 1
    RENDER_MODE_LINENN = 2
    RENDER_MODE_LINEXS = 3
    RENDER_MODE_FIELD = 4
    RENDER_MODE_FIELDLOG = 5
    RENDER_MODE_FIELDSINC = 6
    def __init__(
            self,
            root: str,
            loader: Callable[[str], Any] = default_loader,
            extensions: Optional[Tuple[str, ...]] = IMG_EXTENSIONS,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            is_valid_file: Optional[Callable[[str], bool]] = None,
            width:int = 512, height:int = 512,
            npts:int = 100000, drift_weight:float = 0.4, render_mode:int = 0,
            patch_mode:int = -1, patch_num:int = 10,
            patchgen_seed:int = 100, pointgen_seed:int = 100,
    ) -> None:
        super(datasets.DatasetFolder, self).__init__(root, transform=transform,
                                            target_transform=target_transform)

        self.loader = loader
        self.extensions = extensions

        #self.batch_size:int = batch_size
        self.width:int = width
        self.height:int = height
        self.npts:int = npts
        self.clist:List[int,str,List[List[float]]] = None
        self.ilist:List[int,str,List[List[float]]] = None
        self.ncategory:int = 0
        self.ninstance:int = 0
        self.transform:torchvision.transforms = transform
        self.drift_weight:float = drift_weight
        self.batch_num:int = 0
        self.render_mode:int = render_mode
        self.patch_mode:int = patch_mode
        self.patchgen_seed:int = patchgen_seed
        #self.patchgen_rng:np.random.Generator = None
        self.patchgen_rng = np.random.default_rng(1)
        self.pointgen_seed:int = pointgen_seed
        self.patch_num:int = patch_num

        self.load_categories(self.root)

        self.generate_instances()

        self.ninstance = len(self.ilist)*4*10 # drifting * flip_patterns * patch_patterns
        #self.batch_num = int(math.ceil(self.ninstance/self.batch_size))
    
    def __len__(self) -> int:
        return self.ninstance

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        
        sample, target =  self.get_instance(index)
        
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        
        return sample, target
       

    def load_categories(self, param_dir:str) -> None:
        """load category params from csv
        """
        self.clist = []
        ocdns = os.listdir(param_dir)
        print(ocdns)
        i=0
        for ocdn in ocdns:
            cns = os.listdir(os.path.join(param_dir,ocdn))
            for cn in cns:
                cn_abs = os.path.join(param_dir,ocdn,cn)
                nmaps = np.loadtxt(cn_abs, delimiter=',', dtype=float).reshape((-1,7))
                maps = nmaps.tolist()
                self.clist.append([i, cn, maps])
                i += 1
        self.nclasses = len(self.clist)

    def generate_instances(self) -> None:
        """generate instance params by drifting value of category params
        """
        self.ilist=[]
        _ks=[-2,-1,1,2]
        for ci in range(len(self.clist)):
            _ms=self.clist[ci][2]
            self.ilist.append([self.clist[ci][0], self.clist[ci][1], _ms]) #no drift
            for pi in range(6):
                for k in _ks:
                    __ms=np.array(_ms)
                    __ms[:,pi] *= 1.0+k*self.drift_weight
                    self.ilist.append([self.clist[ci][0], self.clist[ci][1], __ms.tolist()]) #drifted

    def get_instance(self,idx:int) -> Tuple[Any, Any]:
        """get instance image and label
        """

        sidx:int = idx%self.ninstance
        nilist:int = len(self.ilist)
        # iid:int = sidx%nilist
        # flip_flg:int = (sidx//nilist)%4
        # patch_id:int = (sidx//nilist//4)%self.patch_num
        iid:int = (sidx//4//self.patch_num)%nilist
        flip_flg:int = sidx%4
        patch_id:int = (sidx//4)%self.patch_num

        out_label:int = self.ilist[iid][0]
        _mapss:List[List[List[int]]] = [self.ilist[iid][2]]

        pts:np.array = fr.generate(self.npts, _mapss, self.pointgen_seed)
        _imgs:torch.LongTensor = None
        if self.render_mode == self.RENDER_MODE_POINT:
            _imgs= fr.render(pts, self.width, self.height, self.patch_mode, flip_flg, self.patchgen_rng.integers(np.iinfo(np.uint32).max))
        elif self.render_mode == self.RENDER_MODE_LINE:
            _imgs= fr.render_lines(pts, self.width, self.height, flip_flg)
        elif self.render_mode == self.RENDER_MODE_LINENN:
            _imgs= fr.render_linesNN(pts, self.width, self.height, flip_flg)
        elif self.render_mode == self.RENDER_MODE_LINEXS:
            _imgs= fr.render_linesXS(pts, self.width, self.height, flip_flg)
        elif self.render_mode == self.RENDER_MODE_FIELD:
            _imgs= fr.render_field(pts, self.width, self.height, flip_flg)
        elif self.render_mode == self.RENDER_MODE_FIELDLOG:
            _imgs= fr.render_fieldLog(pts, self.width, self.height, flip_flg)
        elif self.render_mode == self.RENDER_MODE_FIELDSINC:
            _imgs= fr.render_fieldSinc(pts, self.width, self.height, flip_flg)
        
        # img:torch.LongTensor = _imgs[0].permute(2,0,1) #chw
        out_data = _imgs[0].permute(2,0,1) #chw
        out_data = transforms.ToPILImage()(out_data.squeeze_(0))
        # if self.transform:
        #     out_data = self.transform(img) #chw
        # else:
        #     out_data = _imgs.float()/255.0
        
        return out_data, out_label
