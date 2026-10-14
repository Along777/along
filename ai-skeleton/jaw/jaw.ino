// Mister Bones' jaw. The porch laptop streams servo angles over USB, one byte each:
//   0-180  move the jaw servo to that angle
//   254    hello: replies "JAW1" (how the laptop finds this board among its USB ports)
//   255    relax: let go of the servo (silent; a rubber band holds the jaw shut)
// If the laptop goes quiet for half a second, the servo relaxes on its own.
// Calibration (which angles mean open and closed) lives on the laptop, so this never needs re-flashing.
#include <Servo.h>

const int SERVO_PIN = 9;          // orange servo wire -> D9
const unsigned long QUIET_MS = 500;

Servo jaw;
bool attached = false;
unsigned long lastByte = 0;

void relax() {
  if (attached) {
    jaw.detach();
    attached = false;
  }
}

void setup() {
  Serial.begin(115200);
}

void loop() {
  while (Serial.available() > 0) {
    int b = Serial.read();
    lastByte = millis();
    if (b == 254) {
      Serial.println("JAW1");
    } else if (b == 255) {
      relax();
    } else if (b >= 0 && b <= 180) {
      if (!attached) {
        jaw.attach(SERVO_PIN);
        attached = true;
      }
      jaw.write(b);
    }
  }
  if (attached && millis() - lastByte > QUIET_MS) {
    relax();
  }
}
