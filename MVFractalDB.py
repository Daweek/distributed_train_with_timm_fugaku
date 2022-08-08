import torch
import torchvision
import numpy as np
import re
import math
from PIL import Image
from typing import List, Tuple, Sequence
from torchvision import datasets, transforms
from typing import Any, Callable, cast, Dict, List, Optional, Tuple

import sys
sys.path.append("./build")
from PyMVFractalDBGenerator import PyMVFractalDBGenerator as MVG

## !!!!definitely required!!!!
def worker_init_mvfdb_fn(worker_id : int):
    info = torch.utils.data.get_worker_info()
    # info.id
    # info.num_workers
    # info.seed
    # info.dataset
    np.random.seed(info.seed%(2**32))
    info.dataset.mvg.setViewpointRNGSeed(info.dataset.viewpoint_random_seed + worker_id)
    # print(type(info.dataset.viewpoint_random_skip_iteration))
    info.dataset.mvg.skipViewpointRNG(info.dataset.viewpoint_random_skip_iteration)
    info.dataset.mvg.start(worker_id)
    print("FROM MVFRACTAL")

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

class MVFractalDB(datasets.DatasetFolder):
    def __init__(
        self,
        root: str,
        loader: Callable[[str], Any] = default_loader,
        extensions: Optional[Tuple[str, ...]] = IMG_EXTENSIONS,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        is_valid_file: Optional[Callable[[str], bool]] = None,
        npts : int = 100000,
        paramtrans_grid : int = 3, 
        paramtrans_coeff : float = 0.1,
        image_width : int = 362,
        image_height : int = 362,
        nvi : int = 40,
        random_viewpoint : bool = True,
        pointgen_seed : int = 100,
        viewpoint_distance_min : float = 10.0,
        viewpoint_distance_max : float = 10.0,
        viewpoint_rng_state : bytes = b"",
        viewpoint_random_seed : int = 100,
        viewpoint_random_skip_iteration : int = 0,
        fovy : float = math.pi/4.0,
        znear : float = 0.01,
        zfar : float = 50.0,
        view_mats : List[np.ndarray] = [],
    ) -> None:
        super(datasets.DatasetFolder, self).__init__(root, transform=transform,
                                            target_transform=target_transform)
        """
        params_file : csv file name of parameters of categories
        npts : number of points to generate
        paramtrans_grid : grid size of parameter trans for instance generation from category
        paramtrans_coeff : grid width of parameter trans for instance generation from category
        image_width : image width to render
        image_height : image height to render
        nvi : number of viewpoints
        random_viewpoint : flag whether using random viewpoint
        pointgen_seed : seed of randomizer for point generation
        viewpoint_distance_min : lower bound of distance from object for random viewpoint generation
        viewpoint_distance_max : upper bound of distance from object for random viewpoint generation
        viewpoint_rng_state : initial state of randomiser of viewpoint generation
        viewpoint_random_seed : seed of randomiser of viewpoint generation
        viewpoint_random_skip_iteration : skip count in iteration for randomizer of viewpoint generation
        fovy : vertical field of view in radian
        znear : lower bound of distance of visible area
        zfar : upper bound of distance of visible area
        view_mats : predefinded view matrix (required if random_viewpoint==False)
        """

        # size of grid of parameter to generate instances from category (should be odd)
        self.paramtrans_grid : int = paramtrans_grid
        # width of grid of parameter to generate instances from category
        self.paramtrans_coeff : float = paramtrans_coeff
        # list of tuple of (category id, category name, maps of category)
        self.mapss_cat : List[Tuple[int, str, Sequence[Sequence[float]]]] = []
        # list of tuple of (category id, category name, maps of instance)
        self.mapss_ins : List[Tuple[int, str, Sequence[Sequence[float]]]] = []
        # read parameter file and store in self.mapss_ins
        self.readParamFile(self.root)
        # number of instances
        self.ninstances : int = len(self.mapss_ins)
        # number of viewpoints
        self.nvi : int = nvi
        # seed of randomizer of viewpoint generator
        self.viewpoint_random_seed : int = viewpoint_random_seed
        # skip count in iteration for randomizer of viewpoint generator
        self.viewpoint_random_skip_iteration : int = viewpoint_random_skip_iteration

        # projection matrix by perspective trasnform
        proj_mat : np.ndarray = MVG.perspective(fovy, image_width/image_height, znear, zfar)

        # instance of MVFractalDB generator
        self.mvg : MVG = MVG(
            npts,
            image_width,
            image_height,
            random_viewpoint,
            proj_mat,
            self.viewpoint_random_seed,
            viewpoint_rng_state,
            viewpoint_distance_min,
            viewpoint_distance_max,
            view_mats,
            pointgen_seed
        )
    
    def __len__(self):
        return self.ninstances*self.nvi

    def __getitem__(self, id: int) -> Tuple[Any, Any]:
        vi = id % self.nvi
        iid = id//self.nvi
        target = self.mapss_ins[iid][0]
        maps = self.mapss_ins[iid][2]
        #print(iid, maps)
        pts = self.mvg.generatePoints(maps)
        #print(pts.shape,pts.max(),pts.min())
        view_mat = self.mvg.getViewMat(vi)
        # H x W x 4
        img = self.mvg.render(pts.astype(np.float32), view_mat)
        # print (type(img))
        # print(img.shape)

        # out_data = img.permute(2,0,1) #chw
        # sample = Image.fromarray(img)
        sample = Image.fromarray(np.uint8(img)).convert('RGB')
        # sample.save("test.png","PNG")
        # print(type(sample))
        # sample = transforms.ToPILImage()(img.squeeze_(0))

        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)


        return  sample, target  
            # sample,target

    def readParamFile(self, fn):
        ## load categories
        self.mapss_cat=[]
        with open(fn,"r") as fp:
            lines=[re.sub("\\s*$","",x) for x in fp.readlines() if not re.match("^\\s*$",x)]
        li=0
        while li<len(lines):
            line = lines[li]
            li+=1
            cols=line.split(',')
            cname=cols[0]
            nmap=int(cols[1])
            maps=[]
            for i in range(nmap):
                line=lines[li]
                li+=1
                maps.append([float(x) for x in line.split(',')])
            self.mapss_cat.append((len(self.mapss_cat),cname,maps))
        print("ncategories",len(self.mapss_cat))
        ## generate instances
        self.mapss_ins=[]
        for ci in range(len(self.mapss_cat)):
            nk = self.paramtrans_grid
            cid = self.mapss_cat[ci][0]
            cname = self.mapss_cat[ci][1]
            maps_cat = self.mapss_cat[ci][2]
            self.mapss_ins.append((cid, cname, np.array(maps_cat).copy().tolist()))
            for pi in range(12):
                maps_ins = np.array(maps_cat).copy().tolist()
                for mi in range(len(maps_cat)):
                    for ki in range(-nk//2,nk//2+1):
                        if ki==0:
                            continue
                        maps_ins[mi][pi] += self.paramtrans_coeff*ki
                self.mapss_ins.append((cid,cname,maps_ins))
        print("ninstances",len(self.mapss_ins))

# if __name__=="__main__":
#     dataset = MVFractalDB(
#         params_file = "./test1k.csv"
#         ,npts = 100000
#         ,paramtrans_grid = 3
#         ,paramtrans_coeff = 0.1
#         ,image_width = 362
#         ,image_height = 362
#         ,nvi = 40
#         ,random_viewpoint = True
#         ,pointgen_seed = 100
#         ,viewpoint_distance_min = 10.0
#         ,viewpoint_distance_max = 13.0
#         ,viewpoint_rng_state = b""
#         ,viewpoint_random_seed = 100
#         ,viewpoint_random_skip_iteration = 0
#         ,fovy = math.pi/4.0
#         ,znear = 0.01
#         ,zfar = 100.0
#         ,view_mats = []
#     )
    # num_workers must be number of available GPUs
    # num_workers = MVG.getGPUNum()
    # use persistent workers
    # dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, num_workers=1, worker_init_fn=worker_init_fn, persistent_workers=True)
    # imgs = torch.ByteTensor([])
    # ptss = torch.FloatTensor([])
    # for i,(l,d,pts) in enumerate(dataloader):
    #     imgs = torch.cat((imgs,d))
    #     ptss = torch.cat((ptss,pts))
    #     #print(i,l.shape,d.shape,d.dtype)
    #     print(imgs.shape, imgs.dtype)
    # mimgs = torchvision.utils.make_grid(imgs.permute(0,3,1,2),nrow=16)
    # Image.fromarray(np.array(mimgs.permute(1,2,0))).save("imgs.png")
    # np.savez("ptss.npz",ptss)
