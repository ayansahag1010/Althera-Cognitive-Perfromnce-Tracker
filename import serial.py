import serial
import csv
import time
from datetime import datetime

# ================== CONFIG ==================
SERIAL_PORT = 'COM5'        # Change if needed
BAUD_RATE = 115200
CSV_FILE = 'sensor_data.csv'
# ============================================

def main():
    print("Opening serial port...")
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    time.sleep(2)  # Allow ESP8266 to reset

    print("Serial port opened successfully.")
    print("Logging started... Press Ctrl+C to stop.\n")

    with open(CSV_FILE, mode='a', newline='') as file:
        writer = csv.writer(file)

        # Write header only if file is empty
        if file.tell() == 0:
            writer.writerow([
                "Timestamp",
                "AX",
                "AY",
                "AZ",
                "Motion",
                "HeartRate",
                "SpO2"
            ])

        try:
            while True:
                raw = ser.readline()
                if not raw:
                    continue

                line = raw.decode('utf-8', errors='ignore').strip()

                if not line.startswith("AX"):
                    continue

                # Parse data
                parts = line.split(',')
                data = {}

                for item in parts:
                    if ':' in item:
                        key, value = item.split(':')
                        data[key] = value

                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

                # Save to CSV
                writer.writerow([
                    timestamp,
                    data.get('AX', ''),
                    data.get('AY', ''),
                    data.get('AZ', ''),
                    data.get('Motion', ''),
                    data.get('HR', ''),
                    data.get('SpO2', '')
                ])
                file.flush()

                # ===== LIVE TERMINAL DISPLAY =====
                print(
                    f"[{timestamp}] "
                    f"AX={data.get('AX')} | "
                    f"AY={data.get('AY')} | "
                    f"AZ={data.get('AZ')} | "
                    f"Motion={data.get('Motion')} | "
                    f"HR={data.get('HR')} bpm | "
                    f"SpO2={data.get('SpO2')} %"
                )

        except KeyboardInterrupt:
            print("\nLogging stopped by user.")

        finally:
            ser.close()
            print("Serial port closed.")

# ================== RUN ==================
if __name__ == "__main__":
    main()
