import torch
import torchvision.models as models

# Tải mô hình ResNet50 với trọng số đã được huấn luyện trước
model = models.resnet50(pretrained=True)  # Hoặc weights=None nếu không cần trọng số pretrained

# Chuyển mô hình sang chế độ đánh giá (evaluation mode)
model.eval()

# Lưu mô hình vào file local
torch.save(model.state_dict(), 'resnet50.pth')