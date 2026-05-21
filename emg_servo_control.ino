#include <Servo.h>

const int emgPin = A0;
const int servoPin = 9;
const int ledPin = 13;

Servo myServo;

void setup() {
  pinMode(ledPin, OUTPUT);
  // 속도를 115200으로 상향 조정
  Serial.begin(115200); 
  myServo.attach(servoPin);
  myServo.write(0);
  
  delay(1000);
  Serial.println("SERVO_SYSTEM_READY");
}

void loop() {
  int emgValue = analogRead(emgPin);
  
  // 파이썬 파서와 일치하는 포맷
  Serial.print("EMG_VAL:");
  Serial.println(emgValue);
  
  if (Serial.available() > 0) {
    String input = Serial.readStringUntil('\n');
    if (input.startsWith("SERVO_DEG:")) {
      int degree = input.substring(10).toInt();
      myServo.write(constrain(degree, 0, 180));
    }
  }
  
  // LED 하트비트
  digitalWrite(ledPin, HIGH);
  delay(50);
  digitalWrite(ledPin, LOW);
  delay(50);
}
