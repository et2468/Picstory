from train_component.data_handler import DataHandler
from train_component.trainer import Trainer
from train_component.vgg16_fine_tuner import VGG16FineTuner
import os

# 설정 값들
DATA_DIR = "data_set"
SAVE_PATH = "vgg_2class.pth"
NUM_EPOCHS = 1
LEARNING_RATE = 0.0001
NUM_CLASSES = 2

def main():
    # 데이터 로더 준비
    data_handler = DataHandler(DATA_DIR)
    dataloaders, dataset_sizes, class_names = data_handler.get_dataloaders()
    
    # 모델 초기화
    model = VGG16FineTuner(num_classes=NUM_CLASSES, learning_rate=LEARNING_RATE)
    
    # 이전에 학습된 가중치 불러오기
    if os.path.exists(SAVE_PATH):
        model.load_weights(SAVE_PATH)
    else:
        print("기존 모델 가중치를 찾을 수 없습니다. 새로운 모델로 훈련을 시작합니다.")

    # 트레이너 초기화 및 훈련 시작
    trainer = Trainer(model, dataloaders, dataset_sizes, 
                      num_epochs=NUM_EPOCHS, save_path=SAVE_PATH)
    trainer.run()

if __name__ == '__main__':
    main()