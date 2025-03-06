from ultralytics.nn.tasks import DetectionModel
from ultralytics.models.yolo.detect import DetectionTrainer

# 콜백 함수 정의: 각 epoch 종료 시 호출되어 성능 지표를 출력합니다.
def log_metrics(trainer):
    # 현재 epoch 번호
    current_epoch = trainer.epoch + 1
    # 손실 값
    loss = trainer.loss
    # 기타 성능 지표 (예: mAP)
    metrics = trainer.metrics
    print(f"Epoch {current_epoch}: Loss = {loss}, Metrics = {metrics}")

# 사용자 정의 트레이너 클래스 정의
class CustomTrainer(DetectionTrainer):
    def get_model(self, cfg=None, weights=None, verbose=True):
        """구성 파일(cfg)과 가중치(weights)를 사용하여 YOLO 모델을 로드합니다."""
        model = DetectionModel(cfg, nc=self.data["nc"], verbose=verbose)
        if weights:
            model.load(weights)
        return model

# 학습 설정
overrides = {
    'model': 'yolov11n_det.yaml',  # 모델 구성 파일
    'data': 'voc.yaml',   # 데이터셋 구성 파일
    'epochs': 50,             # 총 학습 epoch 수
    'batch': 1,              # 배치 크기
    'imgsz': 640,             # 입력 이미지 크기
}

# 트레이너 인스턴스 생성
trainer = CustomTrainer(overrides=overrides)

# 콜백 함수 등록
trainer.add_callback("on_train_epoch_end", log_metrics)

# 학습 시작
trainer.train()