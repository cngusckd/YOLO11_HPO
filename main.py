import optuna
import sys
import logging
import yaml
import numpy as np

from optuna.samplers import GridSampler, TPESampler
from optuna.pruners import HyperbandPruner, SuccessiveHalvingPruner

from ultralytics import YOLO



from ultralytics.nn.tasks import DetectionModel
from ultralytics.models.yolo.detect import DetectionTrainer


class CustomModel(DetectionModel):
    def __init__(self, cfg = 'yolo11n.yaml', ch=3, nc = 20, verbose = True):
        super().__init__()
        """Initialize the YOLO detection model with the given config and parameters."""
        super().__init__()
        self.yaml = cfg if isinstance(cfg, dict) else yaml_model_load(cfg)  # cfg dict
        if self.yaml["backbone"][0][2] == "Silence":
            LOGGER.warning(
                "WARNING ⚠️ YOLOv9 `Silence` module is deprecated in favor of torch.nn.Identity. "
                "Please delete local *.pt file and re-download the latest model checkpoint."
            )
            self.yaml["backbone"][0][2] = "nn.Identity"

        # Define model
        ch = self.yaml["ch"] = self.yaml.get("ch", ch)  # input channels
        if nc and nc != self.yaml["nc"]:
            LOGGER.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml["nc"] = nc  # override YAML value
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=ch, verbose=verbose)  # model, savelist
        self.names = {i: f"{i}" for i in range(self.yaml["nc"])}  # default names dict
        self.inplace = self.yaml.get("inplace", True)
        self.end2end = getattr(self.model[-1], "end2end", False)

        # Build strides
        m = self.model[-1]  # Detect()
        if isinstance(m, Detect):  # includes all Detect subclasses like Segment, Pose, OBB, WorldDetect
            s = 256  # 2x min stride
            m.inplace = self.inplace

            def _forward(x):
                """Performs a forward pass through the model, handling different Detect subclass types accordingly."""
                if self.end2end:
                    return self.forward(x)["one2many"]
                return self.forward(x)[0] if isinstance(m, (Segment, Pose, OBB)) else self.forward(x)

            m.stride = torch.tensor([s / x.shape[-2] for x in _forward(torch.zeros(1, ch, s, s))])  # forward
            self.stride = m.stride
            m.bias_init()  # only run once
        else:
            self.stride = torch.Tensor([32])  # default stride for i.e. RTDETR

        # Init weights, biases
        initialize_weights(self)
        if verbose:
            self.info()
            LOGGER.info("")

class CustomTrainer(DetectionTrainer):
    def get_model(self, cfg, weights):
        """Returns a customized detection model instance configured with specified config and weights."""
        return CustomModel(...)



# 탐색 공간 설정 로드
with open('hpo_search_space.yaml') as f:
    search_space = yaml.safe_load(f)

# HPO 알고리즘 설정 로드
with open('hpo_arguments.yaml') as f:
    args = yaml.safe_load(f)


def objective(trial):
    # 하이퍼파라미터 범위 설정
    lr0 = trial.suggest_float('lr0', 1e-5, 1e-1, log=True)
    momentum = trial.suggest_float('momentum', 0.6, 0.98)
    weight_decay = trial.suggest_float('weight_decay', 0.0, 0.001)
    # batch_size = trial.suggest_int('batch_size', 16, 64, step=16)

    # YOLO 모델 초기화
    model = YOLO('yolov11n_det.yaml')

    step = 0

    # 학습 파라미터 설정
    train_args = {
        'data': 'voc.yaml',  # 데이터셋 구성 파일 경로
        'epochs': 10,            # 총 학습 에포크 수
        'imgsz': 640,            # 입력 이미지 크기
        'lr0': lr0,              # 초기 학습률
        'momentum': momentum,    # 모멘텀
        'device' : [0, 1],
        'val' : False,
        'plots' : False,
        'weight_decay': weight_decay,  # 가중치 감쇠
        'batch': 128,      # 배치 크기
        'project': 'optuna_yolov11',   # 결과 저장 경로
        'name': f'trial_{trial.number}' # 각 실험에 대한 고유 이름
    }

    model.train(**train_args)
    
    return 0.1

if __name__ == '__main__':
    # Sampler 설정
    if args['sampling'] == 'GRID':
        sampler = GridSampler()
    elif args['sampling'] == 'TPE':
        sampler = TPESampler(n_startup_trials = args['n_startup_trials'])
    else:
        sampler = None
    
    # Pruner 설정
    if args['pruning'] == 'Hyperband':
        pruner = HyperbandPruner(
            min_resource = args['min_resource'],
            max_resource = args['n_trial'],
            reduction_factor = args['reduction_factor']
            )
        
    elif args['pruning'] == 'ASH':
        pruner = SuccessiveHalvingPruner(
            min_resource = args['min_resource'],
            reduction_factor = args['reduction_factor']
        )
    else:
        pruner = None

    if sampler == None or pruner == None:
        print('hpo_arguments.yaml 파일을 다시 작성해주세요')
    
    # logger 설정 및 HPO 진행
    optuna.logging.get_logger('optuna').addHandler(logging.StreamHandler(sys.stdout))
    study = optuna.create_study(
        sampler = sampler,
        pruner = pruner,
        direction = 'maximize'
    )

    study.optimize(objective, n_trials = args['n_trial'])
