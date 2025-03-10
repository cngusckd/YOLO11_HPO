import optuna
import sys
import logging
import yaml
import numpy as np

from optuna.samplers import GridSampler
from ultralytics import YOLO

'''
Grid Search를 위한 코드.
HPO 알고리즘에 대한 설정 => 'hpo_arguments.yaml'
Search Space => 'hpo_search_space.yaml'

YOLOV11에 대한 Search Space로 구성되어 있음
'''



# 탐색 공간 설정 로드
with open('hpo_search_space.yaml') as f:
    search_space = yaml.safe_load(f)

# HPO 알고리즘 설정 로드
with open('hpo_arguments.yaml') as f:
    args = yaml.safe_load(f)

def objective(trial):
    
    # YOLO 모델 및 Trainer 생성
    model = YOLO(args['model_yaml_dir'])

    # Trial 생성
    hyperparameters = {key: trial.suggest_float(key, *value) for key, value in search_space.items()}
    for key in hyperparameters.keys():
        hyperparameters[key] = float(hyperparameters[key])

    train_args = {
        'project': 'optuna_yolo',  # Directory to save training results
        'name': f'trial_{trial.number}',  # trail number
        'data': args['dataset_yaml_dir'],  # 데이터 config yaml
        'seed' : args['SEED'], # 공정한 성능 평가를 위해 모델의 seed는 0으로 설정정
        'epochs': args['training_epochs'], # training epochs
        'imgsz' : args['image_resize'], # resize 이미지 크기
        'batch' : args['training_batch'], # batch size, GPU 자원에 맞게 할당
        'device' : args['device_idx'],
        'val' : False, # Validation 옵션 : False로 설정
        'optimizer' : 'adamW', # Optimizer
        'task': args['task'],
        **hyperparameters
    }

    model.train(**train_args)

    metrics = model.val(data = args['dataset_yaml_dir'], imgsz = args['image_resize'], device = args['device_idx'], batch = args['validation_batch'])

    # 주요 메트릭 추출
    metrics = {
        'mAP50': float(metrics.box.map50),
        'mAP50-95': float(metrics.box.map),
        'precision': float(metrics.box.mp),
        'recall': float(metrics.box.mr)
    }


    return metrics['mAP50']

def main(args, search_space):

    sampler = GridSampler(search_space = search_space)
    
    # logger 설정 및 HPO 진행
    optuna.logging.get_logger('optuna').addHandler(logging.StreamHandler(sys.stdout))
    study = optuna.create_study(
        sampler = sampler,
        direction = 'maximize'
    )
    study.optimize(objective, n_trials = args['n_trial'])

    return None



if __name__ == '__main__':

    main(args, search_space)