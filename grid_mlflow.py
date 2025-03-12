import optuna
import sys
import logging
import yaml
import os
import mlflow
import subprocess
subprocess.run(["yolo", "settings", "mlflow=False"], check=True) # 맨 처음 실행 시 YOLO의 자동 MLflow 로깅 기능 비활성화

from optuna.samplers import GridSampler
from ultralytics import YOLO

'''
Grid Search를 위한 코드.
HPO 알고리즘에 대한 설정 => 'hpo_arguments.yaml'
Search Space => 'hpo_search_space.yaml'

YOLOV11에 대한 Search Space로 구성되어 있음
MLflow를 연동하여 모델을 트래킹 및 저장할 수 있도록 함
'''


# MLflow 설정
MLFLOW_TRACKING_URI = "file:./mlruns"  # 로컬 디렉토리에 저장
EXPERIMENT_NAME = "YOLO_HPO"  # MLflow에 기록할 실험 이름

# 탐색 공간 설정 로드
with open('hpo_search_space.yaml', 'r', encoding='utf-8') as f:
    search_space = yaml.safe_load(f)

# HPO 알고리즘 설정 로드
with open('hpo_arguments.yaml', 'r', encoding='utf-8') as f:
    args = yaml.safe_load(f)

def objective(trial):
    # MLflow 실행 시작
    with mlflow.start_run(run_name=f"trial_{trial.number}") as run:
        run_id = run.info.run_id

        # MLflow에 데이터셋 정보 기록
        try:
            # 데이터셋 YAML 파일 읽기
            with open(args['dataset_yaml_dir'], 'r', encoding='utf-8') as f:
                dataset_info = yaml.safe_load(f)
            
            # 데이터셋 입력 정보로 기록
            dataset_path = dataset_info.get('path', 'Unknown')
            dataset_name = os.path.basename(dataset_path) if dataset_path != 'Unknown' else os.path.basename(args['dataset_yaml_dir'])
            
            # MLflow에 데이터셋 입력 정보 기록
            mlflow.log_input(
                mlflow.data.from_dict(
                    {"dataset_yaml": args['dataset_yaml_dir'],
                    "dataset_name": dataset_name,
                    "dataset_path": dataset_path,
                    "train_images": dataset_info.get('train', 'Unknown'),
                    "val_images": dataset_info.get('val', 'Unknown'),
                    "test_images": dataset_info.get('test', 'Unknown'),
                    "nc": dataset_info.get('nc', 'Unknown'),
                    "names": dataset_info.get('names', 'Unknown')
                    },
                    name="YOLO_dataset"
                )
            )
            
            # 데이터셋 YAML 파일 자체를 아티팩트로 저장
            mlflow.log_artifact(args['dataset_yaml_dir'], "dataset_config")
        except Exception as e:
            print(f"데이터셋 정보 기록 중 오류 발생: {e}")


        # YOLO 모델 및 Trainer 생성
        model = YOLO(args['model_yaml_dir'])

        # Trial 생성
        hyperparameters = {key: trial.suggest_float(key, *value) for key, value in search_space.items()}
        for key in hyperparameters.keys():
            hyperparameters[key] = float(hyperparameters[key])

        # 모든 하이퍼파라미터 로깅
        mlflow.log_params(hyperparameters)

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

        # 메트릭 로깅
        mlflow.log_metrics(metrics)
        
        results_path = os.path.join("optuna_yolo", f"trial_{trial.number}")
        
        if os.path.exists(results_path):
            # 모델 저장 (artifacts)
            model_path = os.path.join(results_path, "weights", "best.pt")
            if os.path.exists(model_path):
                mlflow.log_artifact(model_path, "model")
                
            # 훈련 및 검증 그래프 저장 (artifacts)
            for file in os.listdir(results_path):
                if file.endswith(('.png', '.jpg', '.jpeg')) or file == 'results.csv':
                    file_path = os.path.join(results_path, file)
                    if os.path.exists(file_path):
                        mlflow.log_artifact(file_path, "results")        
        else:
            print(f"경고: 결과 경로가 존재하지 않습니다: {results_path}")   
        

    return metrics['mAP50']

def main(args, search_space):    
    # MLflow 설정
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    
    # 실험 생성 또는 가져오기
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        experiment_id = mlflow.create_experiment(
            EXPERIMENT_NAME,
            tags={"search_method": "grid_search", "model": "YOLOV11"}
        )
    else:
        experiment_id = experiment.experiment_id
        
    mlflow.set_experiment(EXPERIMENT_NAME)
    
    # 실험 구성 로깅
    with mlflow.start_run(run_name="hpo_configuration") as parent_run:
        # 검색 공간 및 HPO 설정 로깅
        mlflow.log_param("search_space", search_space)
        mlflow.log_param("n_trials", args['n_trial'])
        mlflow.log_param("optimization_direction", "maximize")
        mlflow.log_param("target_metric", "mAP50")
        
        # HPO arguments YAML 저장
        mlflow.log_artifact('hpo_arguments.yaml', "config")
        mlflow.log_artifact('hpo_search_space.yaml', "config")
        
        
    # sampler 설정
    sampler = GridSampler(search_space = search_space)
    
    # logger 설정 및 HPO 진행
    optuna.logging.get_logger('optuna').addHandler(logging.StreamHandler(sys.stdout))
    study = optuna.create_study(
        sampler = sampler,
        direction = 'maximize'
    )
    study.optimize(objective, n_trials = args['n_trial'])
    
    # 최적 trial 정보 출력
    best_trial = study.best_trial
    print(f"\n최적 Trial 번호: {best_trial.number}")
    print(f"최적 mAP50 값: {best_trial.value}")
    print(f"최적 하이퍼파라미터:")
    for key, value in best_trial.params.items():
        print(f"  {key}: {value}")

    return None



if __name__ == '__main__':

    # MLflow UI 접근 방법 안내
    print(f"\nMLflow 대시보드를 확인하려면 다음 명령어를 실행하세요:")
    print(f"mlflow ui --backend-store-uri {MLFLOW_TRACKING_URI}")
    print(f"원격 서버에서 실행 중인 경우 포트 포워딩을 사용할 수 있습니다:")
    print(f"ssh -L 5000:localhost:5000 username@server_address")

    main(args, search_space)