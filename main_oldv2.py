import optuna
import sys
import logging
import yaml
import numpy as np

from optuna.samplers import GridSampler, TPESampler
from optuna.pruners import HyperbandPruner, SuccessiveHalvingPruner
from ultralytics import YOLO

from Trainer import HPO_Trainer

# 탐색 공간 설정 로드
with open('hpo_search_space.yaml') as f:
    search_space = yaml.safe_load(f)

# HPO 알고리즘 설정 로드
with open('hpo_arguments.yaml') as f:
    args = yaml.safe_load(f)

# TPE, PRuning => 매 epoch마다 학습 => 추론 => pruning
## 1 epoch 학습 시키고 여러가지 파라미터에 대해서 => 성능지표로 pruning
## n epoch 학습시킨 결과들을가지고 => sampling

def objective(trial):
    
    # YOLO 모델 및 Trainer 생성
    model = YOLO('yolov11n_det.yaml')

    # Trainer = HPO_Trainer(model)

    # Trial 생성
    hyperparameters = {key: trial.suggest_float(key, *value) for key, value in search_space.items()}
    for key in hyperparameters.keys():
        hyperparameters[key] = float(hyperparameters[key])

    train_args = {
        'project': 'optuna_yolo',  # Directory to save training results
        'name': f'trial_{trial.number}',  # trail number
        'data': 'voc.yaml',  # 데이터 config yaml
        'seed' : 0, # 공정한 성능 평가를 위해 모델의 seed는 0으로 설정정
        'epochs': 1,
        'imgsz' : 160,
        'batch' : 256,
        'device' : [0, 1],
        'val' : False,
        'lr0' : hyperparameters['lr0'],
        'lrf' : hyperparameters['lrf'],
        'momentum' : hyperparameters['momentum'],
        'optimizer' : 'adamW'
    }

    for step in range(args['n_train_iter']):
        model.train(**train_args)

        metrics = model.val(data = 'voc.yaml', imgsz = 160, device = [0, 1], batch = 256)
    
        # 주요 메트릭 추출
        metrics = {
            'mAP50': float(metrics.box.map50),
            'mAP50-95': float(metrics.box.map),
            'precision': float(metrics.box.mp),
            'recall': float(metrics.box.mr)
        }

        trial.report(metrics['mAP50'], step)
        if trial.should_prune():
            print('this trial is pruned!')
            raise optuna.TrialPruned()

    return metrics['mAP50']

def main(args, search_space):

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

    return None



if __name__ == '__main__':

    main(args, search_space)