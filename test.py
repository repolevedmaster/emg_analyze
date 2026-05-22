import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder

# =========================================================
# 개념
# =========================================================
# 가상 근전도(EMG) 신호 →
# AI semantic decoder →
# 텍스트 단어 출력
#
# 즉:
#
# EMG -> TEXT
#
# 사람의 생각/발화 이전 근육 패턴을
# AI가 의미 벡터로 해석하는 구조.
# =========================================================


# =========================================================
# 가상 단어 사전
# =========================================================
WORDS = [
    "hello",
    "water",
    "computer",
    "open",
    "close",
    "yes",
    "no",
    "help"
]


# =========================================================
# 가상 EMG 생성
# =========================================================
# 각 단어마다
# 고유한 근육 activation 패턴 생성
# =========================================================

SAMPLES_PER_WORD = 200

TIME_STEPS = 128
CHANNELS = 8

X = []
Y = []

np.random.seed(42)

for idx, word in enumerate(WORDS):

    # 단어별 고유 semantic pattern
    base_pattern = np.random.randn(
        CHANNELS
    ) * (idx + 1)

    for _ in range(SAMPLES_PER_WORD):

        signal = []

        for t in range(TIME_STEPS):

            # 시간 변화 + 노이즈
            temporal = np.sin(
                t / 10 + idx
            )

            noise = np.random.randn(
                CHANNELS
            ) * 0.2

            emg = (
                base_pattern * temporal
            ) + noise

            signal.append(emg)

        signal = np.array(signal)

        # 정규화
        signal = (
            signal - np.mean(signal)
        ) / (
            np.std(signal) + 1e-8
        )

        X.append(signal)
        Y.append(word)

X = np.array(X)

print("EMG Shape:", X.shape)


# =========================================================
# 라벨 인코딩
# =========================================================
encoder = LabelEncoder()

y = encoder.fit_transform(Y)

X = torch.tensor(X).float()
y = torch.tensor(y).long()


# =========================================================
# EMG -> TEXT Translator
# =========================================================
class EMGTranslator(nn.Module):

    def __init__(self, vocab_size):

        super().__init__()

        # EMG Temporal Encoder
        self.encoder = nn.Sequential(

            nn.Conv1d(
                CHANNELS,
                32,
                kernel_size=5,
                padding=2
            ),

            nn.ReLU(),

            nn.Conv1d(
                32,
                64,
                kernel_size=5,
                padding=2
            ),

            nn.ReLU(),

            nn.Conv1d(
                64,
                128,
                kernel_size=5,
                padding=2
            ),

            nn.ReLU(),

            nn.AdaptiveAvgPool1d(16)
        )

        # Semantic Latent Space
        self.semantic = nn.Sequential(

            nn.Flatten(),

            nn.Linear(
                128 * 16,
                256
            ),

            nn.ReLU(),

            nn.Linear(
                256,
                128
            ),

            nn.ReLU()
        )

        # Text Decoder
        self.decoder = nn.Linear(
            128,
            vocab_size
        )

    def forward(self, x):

        # [B, T, C] -> [B, C, T]
        x = x.permute(0, 2, 1)

        latent = self.encoder(x)

        semantic = self.semantic(latent)

        text_logits = self.decoder(semantic)

        return text_logits, semantic


# =========================================================
# 모델 생성
# =========================================================
model = EMGTranslator(
    vocab_size=len(WORDS)
)

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)


# =========================================================
# 학습
# =========================================================
EPOCHS = 30

for epoch in range(EPOCHS):

    model.train()

    output, semantic = model(X)

    loss = criterion(output, y)

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    pred = torch.argmax(output, dim=1)

    acc = (pred == y).float().mean()

    print(
        f"epoch {epoch+1}",
        "loss:",
        loss.item(),
        "acc:",
        acc.item()
    )


# =========================================================
# 테스트
# =========================================================
model.eval()

with torch.no_grad():

    idx = np.random.randint(0, len(X))

    sample = X[idx:idx+1]

    output, semantic = model(sample)

    pred = torch.argmax(output, dim=1)

    predicted_word = encoder.inverse_transform(
        [pred.item()]
    )[0]

    real_word = encoder.inverse_transform(
        [y[idx].item()]
    )[0]

    print("\n====================")
    print("REAL WORD:", real_word)
    print("PRED WORD:", predicted_word)
    print("====================")


# =========================================================
# Semantic Synchronization
# =========================================================
# semantic 벡터 =
# AI 내부 의미공간
#
# 이 공간이:
#
# 사람의 근전도 패턴 의미
# 와
# AI 언어 의미
#
# 를 연결하는 중간 계층 역할.
# =========================================================

print("\nSemantic Vector Shape:")
print(semantic.shape)
