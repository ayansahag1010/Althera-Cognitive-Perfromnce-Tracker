import time
import random
import pyttsx3
import csv
import os
from datetime import datetime


# ---------------- VOICE FUNCTION ----------------
def speak(text):
    print("[VOICE]:", text)
    engine = pyttsx3.init()
    engine.setProperty('rate', 135)
    engine.setProperty('volume', 1.0)
    engine.say(text)
    engine.runAndWait()
    engine.stop()
    time.sleep(0.6)


# ---------------- SAVE REACTION RESULTS ----------------
def save_reaction(simple=None, choice=None, accuracy=None, taps=None):

    filename = "reaction_results.csv"
    file_exists = os.path.isfile(filename)

    with open(filename, "a", newline="") as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow([
                "timestamp",
                "simple_reaction_ms",
                "choice_reaction_ms",
                "choice_accuracy_percent",
                "finger_taps"
            ])

        writer.writerow([
            datetime.now(),
            simple,
            choice,
            accuracy,
            taps
        ])


# ---------------- SIMPLE REACTION TEST ----------------
def simple_visual_reaction():
    speak("Simple visual reaction test selected.")
    speak("One practice trial followed by three recorded trials.")

    print("\nPRACTICE TRIAL")
    speak("Practice trial. Wait for go.")
    time.sleep(random.uniform(2, 4))
    print("GO!")
    input()

    speak("Recorded trials starting.")

    reaction_times = []

    for i in range(3):
        speak(f"Trial {i+1}")
        time.sleep(random.uniform(2, 5))
        print("GO!")

        start = time.time()
        input()
        end = time.time()

        rt = round((end - start) * 1000, 2)
        reaction_times.append(rt)

        speak(f"Reaction time {rt} milliseconds.")

    avg_rt = round(sum(reaction_times) / len(reaction_times), 2)

    save_reaction(simple=avg_rt)

    speak(f"Your average reaction time is {avg_rt} milliseconds.")
    return avg_rt


# ---------------- CHOICE REACTION TEST ----------------
def choice_reaction_test():

    speak("Choice reaction test selected.")

    print("\nPRACTICE TRIAL")
    speak("Practice trial. Press R for red or G for green.")

    print(random.choice(["RED", "GREEN"]))
    input()

    reaction_times = []
    correct = 0

    for i in range(3):

        color = random.choice(["RED", "GREEN"])
        print(color)

        start = time.time()
        response = input("Your response (R/G): ").strip().lower()
        end = time.time()

        rt = round((end - start) * 1000, 2)
        reaction_times.append(rt)

        if (color == "RED" and response == "r") or (color == "GREEN" and response == "g"):
            correct += 1

        speak(f"Reaction time {rt} milliseconds.")

    avg_rt = round(sum(reaction_times) / len(reaction_times), 2)
    accuracy = round((correct / 3) * 100, 2)

    save_reaction(choice=avg_rt, accuracy=accuracy)

    speak(f"Average reaction time {avg_rt} milliseconds.")
    speak(f"Accuracy {accuracy} percent.")

    return avg_rt, accuracy


# ---------------- FINGER TAPPING TEST ----------------
def finger_tapping_test():

    speak("Finger tapping test selected.")

    print("\nPRACTICE TRIAL")
    speak("Practice tapping.")

    start = time.time()
    while time.time() - start < 3:
        input()

    tap_counts = []

    for i in range(3):

        speak(f"Trial {i+1}. Tap enter repeatedly for five seconds.")
        print("Tap ENTER repeatedly.")

        count = 0
        start = time.time()

        while time.time() - start < 5:
            input()
            count += 1

        tap_counts.append(count)
        speak(f"You tapped {count} times.")

    avg_taps = round(sum(tap_counts) / len(tap_counts), 2)

    save_reaction(taps=avg_taps)

    speak(f"Average tapping speed {avg_taps} taps.")
    return avg_taps


# ---------------- MAIN MENU ----------------
def main():

    speak("Reaction test module started.")

    while True:

        print("\nChoose a test:")
        print("1. Simple Reaction Test")
        print("2. Choice Reaction Test")
        print("3. Finger Tapping Test")
        print("4. Exit")

        choice = input("Enter choice: ")

        if choice == "1":
            simple_visual_reaction()

        elif choice == "2":
            choice_reaction_test()

        elif choice == "3":
            finger_tapping_test()

        elif choice == "4":
            speak("Exiting reaction module.")
            break


if __name__ == "__main__":
    main()