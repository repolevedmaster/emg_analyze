import serial
import serial.tools.list_ports
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import torch
import torch.nn as nn
import torch.optim as optim
import threading
import time

# ==========================================
# 1. 전역 설정 및 변수
# ==========================================
def find_arduino_port():
    ports = list(serial.tools.list_ports.comports())
    for p in ports:
        if 'Arduino' in p.description or 'USB Serial' in p.description:
            return p.device
    return 'COM3' 

PORT = find_arduino_port() 
BAUD = 115200 
WINDOW_SIZE = 1000 
SAMPLING_LEN = 250 
CONTROL_THRESHOLD = 0.6 
INTENT_CLASSES = ["REST", "MOVE", "STOP"] # 3단계 의도 클래스

class EMGDataBuffer:
    def __init__(self, size):
        self.data = np.zeros(size)
        self.is_running = True
        self.count = 0
        self.current_servo_pos = 0
        
    def update(self, val):
        self.data = np.roll(self.data, -1)
        self.data[-1] = val
        self.count += 1

buffer = EMGDataBuffer(WINDOW_SIZE)

# ==========================================
# 2. NSA(Neural Semantic Alignment) 기반 운동 의도 모델
# ==========================================
class EMGMotionAligner(nn.Module):
    def __init__(self, latent_dim=256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(SAMPLING_LEN, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Linear(512, latent_dim),
            nn.Tanh() # 의도 공간 정렬
        )
        self.classifier = nn.Linear(latent_dim, len(INTENT_CLASSES))
        
    def forward(self, x, return_latent=False):
        latent = self.encoder(x)
        if return_latent:
            return latent
        return self.classifier(latent)

model = EMGMotionAligner()
optimizer = optim.AdamW(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

# ==========================================
# 3. 직렬 통신 (양방향: EMG 수신 & Servo 송신)
# ==========================================
ser_obj = None

def serial_worker():
    global ser_obj
    print(f"연결 시도 중: {PORT} (Baud: {BAUD})")
    try:
        ser_obj = serial.Serial(PORT, BAUD, timeout=0.1)
        time.sleep(2)
        print("성공: 하드웨어 양방향 통신 시작.")
        
        while buffer.is_running:
            if ser_obj.in_waiting:
                line = ser_obj.readline().decode('utf-8', errors='ignore').strip()
                if "EMG_VAL:" in line:
                    try:
                        val = int(line.split("EMG_VAL:")[1])
                        buffer.update(val)
                    except: pass
    except Exception as e:
        print(f"통신 오류: {e}")

def send_servo_command(angle):
    if ser_obj and ser_obj.is_open:
        cmd = f"SERVO_DEG:{angle}\n"
        ser_obj.write(cmd.encode())

# ==========================================
# 4. 운동 의도 학습 및 실시간 제어
# ==========================================
def start_motion_calibration():
    print("\n[단계 1] 3단계 움직임 의도 정밀 캘리브레이션 시작 (10회 분할 샘플링)")
    training_data = []
    training_labels = []
    
    time.sleep(1)
    for idx, intent in enumerate(INTENT_CLASSES):
        print(f"\n>>> '{intent}' 의도 학습을 시작합니다.")
        if intent == "REST":
            state = "편하게 이완하세요 (기본 상태)"
        elif intent == "MOVE":
            state = "서보를 움직이겠다는 강한 의지를 가지세요"
        else: # STOP
            state = "현재 위치에서 멈춰 고정하겠다는 의지를 가지세요"
            
        for s in range(10):
            input(f"[{intent}] {s+1}/10회차: {state} 상태에서 Enter...")
            time.sleep(0.2)
            window = buffer.data[-SAMPLING_LEN:]
            window_norm = (window - np.mean(window)) / (np.std(window) + 1e-8)
            training_data.append(window_norm)
            training_labels.append(idx)
            print(f"   -> {s+1}/10 수집 완료")

    X = torch.FloatTensor(np.array(training_data))
    y = torch.LongTensor(np.array(training_labels))
    
    print("\n3단계 움직임 의도 공간 최적화 중...")
    model.train()
    for epoch in range(300):
        optimizer.zero_grad()
        loss = criterion(model(X), y)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 50 == 0:
            print(f"   Epoch {epoch+1}/300 | Loss: {loss.item():.4f}")
            
    model.eval()
    print("정밀 캘리브레이션 완료.")

def continuous_motion_control():
    print("\n[단계 2] 실시간 3단계 의도 제어 활성화")
    servo_angle = 0
    
    while buffer.is_running:
        if buffer.count < SAMPLING_LEN:
            time.sleep(0.1)
            continue
            
        window = buffer.data[-SAMPLING_LEN:]
        window_norm = (window - np.mean(window)) / (np.std(window) + 1e-8)
        sample = torch.FloatTensor(window_norm).unsqueeze(0)
        
        with torch.no_grad():
            logits = model(sample)
            probs = torch.softmax(logits, dim=1)
            conf, pred = torch.max(probs, dim=1)
            detected_intent = INTENT_CLASSES[pred.item()]
            
            if conf.item() > CONTROL_THRESHOLD:
                if detected_intent == "MOVE":
                    servo_angle = min(180, servo_angle + 8)
                    status = "MOVING (+)"
                elif detected_intent == "STOP":
                    # 현재 각도 유지 (변경 없음)
                    status = "STOPPED (HOLD)"
                else: # REST
                    servo_angle = max(0, servo_angle - 5)
                    status = "RESTING (-)"
                
                print(f"\r[Intent: {detected_intent:4s}] {status:12s} | Conf: {conf.item():.2f} | Angle: {servo_angle:3d}", end="")
            
            send_servo_command(servo_angle)
            
        time.sleep(0.05)

# ==========================================
# 5. 시각화 및 실행
# ==========================================
fig, ax = plt.subplots(figsize=(10, 4))
line, = ax.plot(buffer.data)
ax.set_title("Real-time EMG Intent Visualization")
ax.set_ylim(0, 1024)

def animate(i):
    line.set_ydata(buffer.data)
    return line,

if __name__ == "__main__":
    t1 = threading.Thread(target=serial_worker)
    t1.daemon = True
    t1.start()

    t2 = threading.Thread(target=lambda: (start_motion_calibration(), continuous_motion_control()))
    t2.daemon = True
    t2.start()

    ani = FuncAnimation(fig, animate, interval=30, blit=False)
    plt.tight_layout()
    plt.show()
    buffer.is_running = False
