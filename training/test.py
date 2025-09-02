import torch
import os
from train_component.data_handler import DataHandler
from train_component.vgg16_fine_tuner import VGG16FineTuner

# 설정 값들 (학습 시 사용한 것과 동일해야 함)
DATA_DIR = "data_set"
MODEL_PATH = "vgg_2class.pth"
NUM_CLASSES = 2

def main():
    # 1. 데이터 로더 준비
    data_handler = DataHandler(DATA_DIR)
    dataloaders, dataset_sizes, class_names = data_handler.get_dataloaders()
    test_dataloader = dataloaders["val"]
    test_dataset_size = dataset_sizes["val"]

    # 2. 모델 아키텍처 초기화
    model = VGG16FineTuner(num_classes=NUM_CLASSES)

    # 3. 모델 가중치 로드
    if os.path.exists(MODEL_PATH):
        model.load_weights(MODEL_PATH)
    else:
        print(f"오류: '{MODEL_PATH}' 파일을 찾을 수 없습니다. 테스트를 진행할 수 없습니다.")
        return

    # 4. 모델 테스트 수행
    print("\n모델 테스트를 시작합니다...")
    test_loss, test_acc = model.evaluate_model(
        test_dataloader, desc="테스트 중"
    )

    print("\n--- 테스트 결과 ---")
    print(f"테스트 손실 (Loss): {test_loss:.4f}")
    print(f"테스트 정확도 (Accuracy): {test_acc:.4f}")
    print(
        f"전체 {test_dataset_size}개의 테스트 이미지 중 {int(test_acc * test_dataset_size)}개를 올바르게 분류했습니다."
    )

if __name__ == '__main__':
    main()