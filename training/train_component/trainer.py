class Trainer:
    def __init__(
        self,
        model,
        dataloaders,
        dataset_sizes,
        num_epochs,
        save_path,
    ):
        self.model = model
        self.dataloaders = dataloaders
        self.dataset_sizes = dataset_sizes
        self.num_epochs = num_epochs
        self.save_path = save_path

    def run(self):
        best_acc = 0.0

        for epoch in range(self.num_epochs):
            print(f"Epoch {epoch+1}/{self.num_epochs}")
            print("-" * 30)

            # VGG16FineTuner 클래스의 메서드를 호출합니다.
            train_loss, train_acc = self.model.train_epoch(self.dataloaders["train"])
            print(f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f}")

            val_loss, val_acc = self.model.validate_epoch(self.dataloaders["val"])
            print(f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

            if val_acc > best_acc:
                best_acc = val_acc
                self.model.save_model(self.save_path)
                print(f"Best model saved with accuracy: {best_acc:.4f}")

        print(f"\nTraining complete. Best val accuracy: {best_acc:.4f}")