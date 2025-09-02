import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io
import os

class InferenceModel(nn.Module):
    def __init__(self, model_path, class_names, device):
        super().__init__()
        
        self.model = models.vgg16(weights=None)
        
        self.device = device
        self.model_path = model_path
        self.class_names = class_names
        
        # 이미지 전처리 파이프라인을 정의합니다.
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        
        # 분류기 마지막 층을 출력 클래스 수에 맞게 변경합니다.
        num_ftrs = self.model.classifier[6].in_features
        self.model.classifier[6] = nn.Linear(num_ftrs, len(self.class_names))
        
        # 모델 파일이 존재하면 가중치를 로드합니다.
        if os.path.exists(self.model_path):
            state_dict = torch.load(self.model_path, map_location=self.device)
            # 가중치 키에 'model.' 접두사를 추가하여 로드합니다.
            new_state_dict = {f'model.{k}': v for k, v in state_dict.items()}
            self.load_state_dict(new_state_dict)
            self.to(self.device)
            self.eval() # 모델을 평가 모드로 설정
            print("모델이 성공적으로 로드되었습니다.")
        else:
            print(f"'{self.model_path}' 경로에 모델 파일이 없습니다.")
            
    def forward(self, x):
        return self.model(x)
            
    def predict_image(self, image_bytes):
        try:
            # 바이트에서 이미지를 열고 전처리를 적용합니다.
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # 기울기 계산 없이 예측을 수행합니다.
            with torch.no_grad():
                outputs = self(image_tensor)
                _, preds = torch.max(outputs, 1)
                
            return self.class_names[preds.item()]
        except Exception as e:
            print(f"이미지 예측 중 오류 발생: {e}")
            return None