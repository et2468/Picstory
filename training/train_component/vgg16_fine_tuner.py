import torch
from torchvision import models
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

class VGG16FineTuner:
    def __init__(self, num_classes=2, learning_rate=0.0001, device=None):
        self.device = (
            device
            if device
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = models.vgg16(weights="DEFAULT")

        for param in self.model.features.parameters():
            param.requires_grad = False

        num_ftrs = self.model.classifier[6].in_features
        self.model.classifier[6] = nn.Linear(num_ftrs, num_classes)
        self.model = self.model.to(self.device)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(
            self.model.classifier.parameters(), lr=learning_rate
        )

    def train_epoch(self, dataloader):
        self.model.train()
        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in tqdm(dataloader, desc="Training"):
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = self.criterion(outputs, labels)

            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(dataloader.dataset)
        epoch_acc = running_corrects.double() / len(dataloader.dataset)

        return epoch_loss, epoch_acc.item()

    def validate_epoch(self, dataloader):
        self.model.eval()
        running_loss = 0.0
        running_corrects = 0

        with torch.no_grad():
            for inputs, labels in tqdm(dataloader, desc="Validating"):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = self.criterion(outputs, labels)

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(dataloader.dataset)
        epoch_acc = running_corrects.double() / len(dataloader.dataset)

        return epoch_loss, epoch_acc.item()

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    def load_weights(self, path):
        """저장된 모델의 가중치를 불러옵니다."""
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        print(f"모델 가중치를 '{path}' 에서 성공적으로 로드했습니다.")

    def evaluate_model(self, dataloader, desc="평가"):
        """모델을 평가하고 손실 및 정확도를 반환합니다."""
        self.model.eval()  # 모델을 평가 모드로 설정
        running_loss = 0.0
        running_corrects = 0

        with torch.no_grad():  # 기울기 계산을 비활성화
            for inputs, labels in tqdm(dataloader, desc=desc):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = self.criterion(outputs, labels)

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

        total_loss = running_loss / len(dataloader.dataset)
        total_acc = running_corrects.double() / len(dataloader.dataset)

        return total_loss, total_acc.item()