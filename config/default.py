from yacs.config import CfgNode as CN
import os, argparse
import sys, time, errno
import datetime
import os.path as osp
import ssl


class Config(object):
    def __init__(self):
        self._C = CN()
        # -----------------------------------------------------------------------------
        # SYS
        self._C.SYS=CN()
        self._C.SYS.ROOT_PATH = '/home/wuqi/'  # '/home/dell3080ti_01/'
        self._C.SYS.OUTPUT_DIR="logs"
        self._C.SYS.OUTPUT_DIR_SET=""
        self._C.SYS.DEVICE='cuda'
        self._C.SYS.DEVICE_IDS=[] # [0,1,2,3]
        self._C.SYS.REQUEST_MEM = 9000
        self._C.SYS.REQUEST_NUM = 1
        self._C.SYS.CUDNN = True

        # -----------------------------------------------------------------------------
        # DATASETS
        self._C.DATASETS = CN()
        self._C.DATASETS.NAMES = 'cuhk03' # 'cuhk03','market1501'
        self._C.DATASETS.ROOT_DIR = 'kr/dataset/'
        self._C.DATASETS.CLUSTER_K = -1
        # DATALOADER
        self._C.DATALOADER=CN()
        self._C.DATALOADER.NUM_WORKERS = 8
        self._C.DATALOADER.NUM_INSTANCE = 4 # Number of instance for one batch
        # INPUT
        self._C.INPUT = CN()
        self._C.INPUT.SIZE_TRAIN = [384, 128]
        self._C.INPUT.SIZE_TEST = [384, 128]
        self._C.INPUT.PROB = 0.5
        self._C.INPUT.RE_PROB = 0.5
        self._C.INPUT.PIXEL_MEAN = [0.485, 0.456, 0.406]
        self._C.INPUT.PIXEL_STD = [0.229, 0.224, 0.225]
        self._C.INPUT.PADDING = 10
        self._C.INPUT.FLIP = False
        self._C.INPUT.MISS_VALUE = -1
        self._C.INPUT.REA = True
        # -----------------------------------------------------------------------------
        # MODEL
        self._C.MODEL = CN()
        self._C.MODEL.BACKBONE_NAME = 'resnet50'
        self._C.MODEL.NAME = 'AGW'
        self._C.MODEL.NECK = 'bnneck'
        self._C.MODEL.NECK_FEAT = 'after'
        self._C.MODEL.LAST_STRIDE=1
        self._C.MODEL.PRETRAIN_CHOICE='imagenet'  #  'imagenet' or 'self'
        self._C.MODEL.SELF_PRETRAIN_MODEL='kr/kr1/reid_framework/out/test/checkpoint.pth.tar'
        self._C.MODEL.GEM_POOL = True
        self._C.MODEL.SAVE = False
        self._C.MODEL.RESCA = False
        self._C.MODEL.FDCA_MODE = 'p'
        self._C.MODEL.FDCA_NUM = (1,1,1,1,1)
        self._C.MODEL.CA_BIAS = True

        # -----------------------------------------------------------------------------
        # LOSS
        self._C.LOSS=CN()
        self._C.LOSS.LOSS_TYPE='xent_htri_cent' # {'xent','htri','cent'}
        self._C.LOSS.MARGIN=0.3 # for htri
        self._C.LOSS.ID_LOSS_WEIGHT= 1
        self._C.LOSS.TRI_LOSS_WEIGHT= 1
        self._C.LOSS.CENTER_LOSS_WEIGHT= 0.0005
        self._C.LOSS.POSE_WEIGHT = 1.0
        self._C.LOSS.CONSIST_WEIGHT = 0.001
        self._C.LOSS.LABELSMOOTH = True
        self._C.LOSS.WEIGHT_REGULARIZED_TRIPLET = True
        self._C.LOSS.DIST_TYPE = 'euclidean'  # 'cosine'
        # -----------------------------------------------------------------------------
        # OPTI
        self._C.OPTIM=CN()
        self._C.OPTIM.OPTIMIZER_NAME = "Adam"
        self._C.OPTIM.BASE_LR = 0.00035
        self._C.OPTIM.BIAS_LR_FACTOR = 1
        self._C.OPTIM.MOMENTUM = 0.9
        self._C.OPTIM.WEIGHT_DECAY = 0.0005
        self._C.OPTIM.WEIGHT_DECAY_BIAS = 0.0005
        self._C.OPTIM.CENTER_LR = 0.5
        # -----------------------------------------------------------------------------
        # SCHEDULER
        self._C.SCHEDULER=CN()
        self._C.SCHEDULER.STEPS = [40, 70] # decay step of learning rate
        self._C.SCHEDULER.GAMMA = 0.1 # decay rate of learning rate
        self._C.SCHEDULER.WARMUP_FACTOR = 0.01 # warm up factor
        self._C.SCHEDULER.WARMUP_ITERS = 10 # iterations of warm up, -1代表不预热学习率
        self._C.SCHEDULER.WARMUP_METHOD = "linear" # method of warm up, option: 'constant','linear'
        self._C.SCHEDULER.ETA_MIN = 0  # 波谷值，一般为0
        self._C.SCHEDULER.POWER = 0.6  # PolyLR参数。越接近1，下降曲线越直；越接近0，越弯曲
        # -----------------------------------------------------------------------------
        # TRAIN
        self._C.TRAIN=CN()
        self._C.TRAIN.BATCH_SIZE=64
        self._C.TRAIN.MAX_EPOCHS=120
        self._C.TRAIN.PRINT_FREQ=10 #
        self._C.TRAIN.SEED=1
        # -----------------------------------------------------------------------------
        # TEST
        self._C.TEST=CN()
        self._C.TEST.BATCH_SIZE=128
        self._C.TEST.FEAT_NORM = True
        self._C.TEST.VALID_FREQ=20
        self._C.TEST.START_EPOCH=0
        self._C.TEST.METRIC_MINP = True
        self._C.TEST.RE_RANKING = False
        self._C.TEST.DIST_TYPE = 'euclidean'  # 'cosine'

    def merge_from_file(self, config_yaml):
        if config_yaml != "":
            self._C.merge_from_file(config_yaml)

    def merge_from_list(self, config_list):
        self._C.merge_from_list(config_list)

    def set_root_path(self, root_path):
        self._C.SYS.ROOT_PATH = root_path

    def freeze(self):
        self._C.freeze()

    @property
    def C(self):
        return self._C

    def gpu_num(self):
        return len(self._C.SYS.DEVICE_IDS)


def __getcfg():
    ssl._create_default_https_context = ssl._create_unverified_context
    parser = argparse.ArgumentParser(description="ReID Baseline Training")
    parser.add_argument("--config_file", default="",
                        help="path to config file", type=str)
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)
    args = parser.parse_args()
    cfg = Config()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    cfg.C.INPUT.SIZE_TEST = cfg.C.INPUT.SIZE_TRAIN
    cfg.C.DATASETS.ROOT_DIR = cfg.C.SYS.ROOT_PATH + cfg.C.DATASETS.ROOT_DIR
    cfg.C.MODEL.SELF_PRETRAIN_MODEL = cfg.C.SYS.ROOT_PATH + cfg.C.MODEL.SELF_PRETRAIN_MODEL
    if cfg.C.SYS.OUTPUT_DIR_SET in ["d", "default"]:
        miss_value = cfg.C.INPUT.MISS_VALUE
        cfg.C.SYS.OUTPUT_DIR = 'logs/'+cfg.C.DATASETS.NAMES+'_'+cfg.C.MODEL.NAME+'_'+cfg.C.MODEL.BACKBONE_NAME+\
                               ('' if cfg.C.INPUT.FLIP else '_noflip')+('' if miss_value==-1 else str(miss_value))
    if cfg.C.SYS.DEVICE == 'cpu':
        cfg.C.TEST.VALID_FREQ = 1
        cfg.C.TRAIN.PRINT_FREQ = 1
    # -------------------------------------------------------------------------------
    cfg=cfg.C
    logger = set_logger(cfg)
    print(args)
    print(cfg)
    if 'wuqi' not in cfg.SYS.ROOT_PATH: waiting_gpu(cfg, logger)
    return cfg, args, logger


def waiting_gpu(cfg, logger):
    logger.set_pr_time()
    if cfg.SYS.DEVICE == 'cuda':
        req_mem = cfg.SYS.REQUEST_MEM
        if len(cfg.SYS.DEVICE_IDS) == 0:
            req_num = cfg.SYS.REQUEST_NUM
            cfg.SYS.DEVICE_IDS = allocate_gpu_num(req_mem, gpu_num=req_num, root_path=cfg.SYS.ROOT_PATH)
            assert len(cfg.SYS.DEVICE_IDS) > 0
        else:
            cfg.SYS.DEVICE_IDS = allocate_gpu_id(req_mem, gpu_id=cfg.SYS.DEVICE_IDS)
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, cfg.SYS.DEVICE_IDS))
    logger.set_pr_time(False)
    if cfg.SYS.DEVICE == 'cuda':
        import torch
        print("Using {} GPUS: {}".format(torch.cuda.device_count(), cfg.SYS.DEVICE_IDS))
    else:
        print("Using CPU")
        print('Note: Using device {} will significantly slow down training, and CUDA is highly recommended.'.
              format(cfg.SYS.DEVICE))


def set_logger(cfg):
    now_time = datetime.datetime.now()
    ymd = now_time.strftime("%Y%m%d")
    hms = now_time.strftime("%H%M%S")
    logger = Logger(osp.join(osp.join(cfg.SYS.OUTPUT_DIR, ymd),
                             hms + '_' + cfg.MODEL.NAME + '_' + cfg.MODEL.BACKBONE_NAME + '.log'))
    sys.stdout = logger
    return logger


def allocate_gpu_num(request_mem=8000, gpu_num=1, root_path="/home/dell3080ti_01/", waiting_time=1, interval=0.2):
    mem_threshold = request_mem
    free_time = 0
    if root_path == "/home/dell3080ti_01/":
        all_num = 4
    elif root_path == "/home/wuqi/":
        all_num = 8
    else:
        raise Exception('Invalid root_path')
    assert gpu_num <= all_num
    print('Waiting for available gpu({}*{} MiB)...'.format(gpu_num, mem_threshold))
    while True:
        gpu_status = os.popen('nvidia-smi | grep %').read().split('|')
        avail_mem = []
        all_gpu_id = range(all_num)
        for i in all_gpu_id:
            if i<0:
                avail=0
            else:
                mem_str = gpu_status[4 * i + 2].split('/')
                avail = int(mem_str[1].split('M')[0].strip()) - int(mem_str[0].split('M')[0].strip())
            avail_mem.append(avail)
        gpu_memory_str = 'Available memory: {}'.format(avail_mem)
        time.sleep(interval)
        num=0
        for m in avail_mem:
            if m > mem_threshold: num+=1
        free_time = (free_time + interval) if num>=gpu_num else 0
        if free_time >= waiting_time:
            print(gpu_memory_str)
            print('Get available gpu.')
            sort_index = sorted(range(len(avail_mem)), key=lambda k: avail_mem[k], reverse=True)
            return sort_index[0:gpu_num]


def allocate_gpu_id(request_mem=8000, gpu_id=None, waiting_time=1, interval=0.2):
    mem_threshold = request_mem
    free_time = 0
    # print('Waiting for gpu {}(threshold is {} MiB)...'.format(gpu_id, mem_threshold))
    print('Waiting for gpu {}...'.format(gpu_id))
    while True:
        gpu_status = os.popen('nvidia-smi | grep %').read().split('|')
        avail_mem = []
        for i in gpu_id:
            mem_str = gpu_status[4 * i + 2].split('/')
            avail = int(mem_str[1].split('M')[0].strip()) - int(mem_str[0].split('M')[0].strip())
            avail_mem.append(avail)
        gpu_memory_str = 'Available memory: {}'.format(avail_mem)
        time.sleep(interval)
        ava = True
        for m in avail_mem:
            if m < mem_threshold: ava = False
        free_time = (free_time + interval) if ava else 0
        if free_time >= waiting_time:
            print(gpu_memory_str)
            print('Get available gpu.')
            sort_index = sorted(range(len(avail_mem)), key=lambda k: avail_mem[k], reverse=True)
            res = []
            for ind in sort_index:
                res.append(gpu_id[ind])
            return res


def mkdir_if_missing(directory):
    if not osp.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise


class Logger(object):
    def __init__(self, fpath=None, pr_time=False):
        self.console = sys.stdout
        self.file = None
        self.pr_time = pr_time
        if fpath is not None:
            mkdir_if_missing(os.path.dirname(fpath))
            self.file = open(fpath, 'w')

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        if self.pr_time and msg not in ['', '\n']:
            timeline = time.strftime('[%Y-%m-%d %H:%M:%S] ', time.localtime(time.time()))
            msg = timeline + msg
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()

    def set_pr_time(self, pr_time=True):
        self.pr_time = pr_time


cfg, args, logger = __getcfg()

