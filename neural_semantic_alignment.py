"""
===========================================================
 Neural Semantic Alignment Decoder
===========================================================
이 코드는 인간의 미세 운동(Subvocalization) 신호와
AI 신경망의 의미 상태 공간(Semantic Latent Space) 간의 
구조적 매핑(Alignment) 가설을 테스트합니다.

[구조]
1. EMG Encoder: 시계열 근전도(EMG) 신호에서 잠재 특징(Latent) 추출
2. Semantic Projection: 추출된 특징을 모델의 의미 상태 공간으로 매핑
3. Decoder: 매핑된 잠재 표현을 기반으로 최종 텍스트 분류
===========================================================
"""

import pickle
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# =========================================================
# 1. 설정 및 하이퍼파라미터
# =========================================================
DATA_FILE = "data.pickle"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 100
LR = 1e-3
BATCH = 64
TARGET_LEN = 2000

print(f"사용 장치: {DEVICE}")

# =========================================================
# 2. 데이터 로드 및 전처리
# =========================================================
print("\n데이터 로딩 중...")
with open(DATA_FILE, "rb") as f:
    raw = pickle.load(f)

data = raw["data"]
labels = raw["labels"]

print(f"원본 데이터 수: {len(data)}")

encoder = LabelEncoder()
Y = encoder.fit_transform(labels)
classes = encoder.classes_
NUM_CLASSES = len(classes)

print("\n클래스 매핑:")
for i, c in enumerate(classes):
    print(f"{i:2d} : {c}")

X = []
newY = []

for emg, label in zip(data, Y):
    try:
        emg = np.array(emg).astype(np.float32)
        
        if len(emg.shape) != 2:
            continue
            
        if emg.shape[1] > 8:
            emg = emg[:, :8]
        elif emg.shape[1] < 8:
            pad = np.zeros((emg.shape[0], 8 - emg.shape[1]), dtype=np.float32)
            emg = np.concatenate([emg, pad], axis=1)

        if emg.shape[0] > TARGET_LEN:
            emg = emg[:TARGET_LEN, :]
        elif emg.shape[0] < TARGET_LEN:
            pad = np.zeros((TARGET_LEN - emg.shape[0], 8), dtype=np.float32)
            emg = np.concatenate([emg, pad], axis=0)

        for c in range(8):
            ch_data = emg[:, c]
            emg[:, c] = (ch_data - np.mean(ch_data)) / (np.std(ch_data) + 1e-8)

        emg = emg.T 
        
        X.append(emg)
        newY.append(label)
    except Exception as e:
        pass

X = np.array(X, dtype=np.float32)
Y = np.array(newY, dtype=np.int64)

print(f"\n전처리 완료 샘플 수: {len(X)}")
print(f"입력 텐서 형태: {X.shape}")

X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42, stratify=Y
)

X_train = torch.tensor(X_train)
y_train = torch.tensor(y_train)
X_test = torch.tensor(X_test)
y_test = torch.tensor(y_test)

# =========================================================
# 3. 모델 정의
# =========================================================
class NeuralSemanticAligner(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        
        # 1. EMG Encoder
        self.emg_encoder = nn.Sequential(
            nn.Conv1d(8, 64, kernel_size=15, stride=2, padding=7),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(64, 128, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(128, 256, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        
        # 2. Semantic Projection
        self.semantic_projection = nn.Sequential(
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.GELU()
        )
        
        # 3. Decoder
        self.decoder = nn.Linear(256, num_classes)
        
    def forward(self, x, return_latents=False):
        emg_latent = self.emg_encoder(x).squeeze(-1) 
        semantic_latent = self.semantic_projection(emg_latent)
        output = self.decoder(semantic_latent)
        
        if return_latents:
            return output, emg_latent, semantic_latent
        return output

model = NeuralSemanticAligner(NUM_CLASSES).to(DEVICE)
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss()

# =========================================================
# 4. 모델 학습
# =========================================================
print("\n================================================")
print(" 학습 시작 ")
print("================================================")

for epoch in range(EPOCHS):
    model.train()
    permutation = torch.randperm(len(X_train))
    total_loss = 0
    correct = 0
    
    for i in range(0, len(X_train), BATCH):
        indices = permutation[i:i+BATCH]
        batch_x = X_train[indices].to(DEVICE)
        batch_y = y_train[indices].to(DEVICE)
        
        optimizer.zero_grad()
        
        pred = model(batch_x)
        loss = criterion(pred, batch_y)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        correct += (torch.argmax(pred, dim=1) == batch_y).sum().item()
        
    avg_loss = total_loss / (len(X_train) / BATCH)
    acc = correct / len(X_train) * 100
    
    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(f"Epoch [{epoch+1:3d}/{EPOCHS}] | Loss: {avg_loss:.4f} | Train Acc: {acc:.2f}%")

# =========================================================
# 5. 성능 평가
# =========================================================
print("\n================================================")
print(" 검증 결과 ")
print("================================================")

model.eval()
correct = 0

with torch.no_grad():
    pred = model(X_test.to(DEVICE))
    predicted = torch.argmax(pred, dim=1)
    correct = (predicted.cpu() == y_test).sum().item()

accuracy = correct / len(y_test)
print(f"\n최종 검증 정확도: {accuracy*100:.2f}%")

# =========================================================
# 6. 추론 과정 출력
# =========================================================
print("\n================================================")
print(" 추론 프로세스 시뮬레이션 ")
print("================================================")

with torch.no_grad():
    for i in range(min(5, len(X_test))):
        sample_x = X_test[i:i+1].to(DEVICE)
        real_idx = y_test[i].item()
        
        output, emg_latent, semantic_latent = model(sample_x, return_latents=True)
        pred_idx = torch.argmax(output, dim=1).item()
        
        real_sentence = classes[real_idx]
        predicted_sentence = classes[pred_idx]
        
        print("\n----------------------------------------------------")
        print(f"[테스트 샘플 {i+1}]")
        print("1. 신호 입력 (EMG 2000 frames)")
        time.sleep(0.1)
        
        print(f"2. 인코더 추출 잠재 벡터 차원: {emg_latent.shape[1]}")
        time.sleep(0.1)
        
        print(f"3. 의미 공간 투영 (Semantic Alignment)")
        time.sleep(0.1)
        
        print(f"4. 디코더 텍스트 출력:")
        print(f"   - 실제 클래스: {real_sentence}")
        print(f"   - 예측 클래스: {predicted_sentence}")
        
        if real_idx == pred_idx:
            print("   - 매핑 결과: 일치")
        else:
            print("   - 매핑 결과: 불일치")
            
        time.sleep(0.3)

print("\n작업 완료")
