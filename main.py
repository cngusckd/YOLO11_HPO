import optuna
import sys
import logging
import yaml
import torch.distributed as dist

from optuna.samplers import TPESampler
from optuna.pruners import HyperbandPruner, SuccessiveHalvingPruner

import sys
import os

from custom_util.Model import CustomDetTrainer, CustomSegTrainer


'''
Sampling과 Pruning을 결합한 코드 구현
현재 DDP에서 CustomDetTrainer와 CustomSegTrainer가 불러와지지 않는 오류 존재, 이를 위한 해결 필요
'''


# 탐색 공간 설정 로드
with open('hpo_search_space.yaml') as f:
    search_space = yaml.safe_load(f)

# HPO 알고리즘 설정 로드
with open('hpo_arguments.yaml') as f:
    args = yaml.safe_load(f)

# callback 함수 추가
def custom_val(trainer):
    metrics, fitness = trainer.validate()

    # 주요 메트릭 추출
    metrics = {
        'mAP50': float(metrics['metrics/mAP50(B)']),
        'mAP50-95': float(metrics['metrics/mAP50-95(B)']),
        'precision': float(metrics['metrics/precision(B)']),
        'recall': float(metrics['metrics/recall(B)'])
    }

    trainer.trial.report(metrics['mAP50'], trainer.epoch)
    print(metrics)

    if trainer.trial.should_prune():
        raise optuna.TrialPruned()

def objective(trial):

    # DDP가 활성화된 경우에만 종료 => process id 충돌방지용
    if dist.is_initialized():
        dist.destroy_process_group()


     # Trial 생성
    hyperparameters = {key: trial.suggest_float(key, *value) for key, value in search_space.items()}
    for key in hyperparameters.keys():
        hyperparameters[key] = float(hyperparameters[key])

    train_args = {
        # 'project': 'optuna_yolo',  # Directory to save training results
        'model' : args['model_yaml_dir'],
        'name': f'trial_{trial.number}',  # trail number
        'data': args['dataset_yaml_dir'],  # 데이터 config yaml
        'seed' : args['SEED'], # 공정한 성능 평가를 위해 모델의 seed는 0으로 설정
        'epochs': args['training_epochs'], # training epochs
        'imgsz' : args['image_resize'], # resize 이미지 크기
        'batch' : args['training_batch'], # batch size, GPU 자원에 맞게 할당
        'device' : args['device_idx'],
        'val' : False, # Validation 옵션 : False로 설정
        'workers' : 4, # process 오류방지용, workers를 큰값으로 설정할경우 process id에서 충돌 발생
        'optimizer' : 'adamW', # Optimizer
        'task': args['task'],
        **hyperparameters
    }

    # Task에 맞는 model과 Trainer 선언
    if args['task'] == 'detect':
        trainer = CustomDetTrainer(overrides = train_args, trial = trial)
        print('callback 적용 전')
        trainer.add_callback('on_train_epoch_end', custom_val)
        print('callback 적용 후')
        
    elif args['task'] == 'segment':
        trainer = CustomSegTrainer(overrides = train_args, trial = trial)
        trainer.add_callback('on_train_epoch_end', custom_val)

    # Training
    trainer.train()

    # Last Epoch => validation 진행
    metrics, fitness = trainer.validate()
    
    # 주요 메트릭 추출
    metrics = {
        'mAP50': float(metrics['metrics/mAP50(B)']),
        'mAP50-95': float(metrics['metrics/mAP50-95(B)']),
        'precision': float(metrics['metrics/precision(B)']),
        'recall': float(metrics['metrics/recall(B)'])
    }

    return metrics['mAP50']

def main(args, search_space):

    # Sampler 설정
    if args['sampling'] == 'TPE':
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

    print('PATH', os.getcwd())
    print('PATH', sys.path)
    main(args, search_space)

