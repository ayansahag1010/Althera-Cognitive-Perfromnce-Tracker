import time
import random
import pyttsx3

# -----------------------------------------
# SAFE SPEAK FUNCTION (CRITICAL FIX)
# -----------------------------------------
def speak(text):
    print("[VOICE]:", text)
    engine = pyttsx3.init()
    engine.setProperty('rate', 135)
    engine.setProperty('volume', 1.0)
    engine.say(text)
    engine.runAndWait()
    engine.stop()
    time.sleep(0.6)
# -----------------------------------------


# ---------------- TEST FUNCTIONS ----------------
def simple_visual_reaction():
    speak("Simple visual reaction test selected.")
    speak("You will first do one practice trial.")
    speak("Then three trials will be recorded.")

    print("\nPRACTICE TRIAL (Not recorded)")
    speak("Practice trial. Wait for go.")
    time.sleep(random.uniform(2, 4))
    print("GO!")
    input()

    speak("Practice completed. Recorded trials will start now.")

    reaction_times = []

    for i in range(3):
        speak(f"Trial number {i + 1}. Get ready.")
        time.sleep(random.uniform(2, 5))
        print("GO!")
        start = time.time()
        input()
        end = time.time()

        rt = round((end - start) * 1000, 2)
        reaction_times.append(rt)
        speak(f"Reaction time {rt} milliseconds.")

    avg_rt = round(sum(reaction_times) / len(reaction_times), 2)
    speak(f"Your average reaction time is {avg_rt} milliseconds.")
    return avg_rt


def choice_reaction_test():
    speak("Choice reaction test selected.")
    speak("You will first do one practice trial.")
    speak("Then three trials will be recorded.")

    print("\nPRACTICE TRIAL (Not recorded)")
    speak("Practice trial. Press R for red or G for green.")
    print(random.choice(["RED", "GREEN"]))
    input()

    speak("Practice completed. Recorded trials will start now.")

    reaction_times = []
    correct = 0

    for i in range(3):
        speak(f"Trial number {i + 1}.")
        color = random.choice(["RED", "GREEN"])
        print(color)

        start = time.time()
        response = input("Your response (R/G): ").strip().lower()
        end = time.time()

        rt = round((end - start) * 1000, 2)
        reaction_times.append(rt)

        if (color == "RED" and response == "r") or \
           (color == "GREEN" and response == "g"):
            correct += 1

        speak(f"Reaction time {rt} milliseconds.")

    avg_rt = round(sum(reaction_times) / len(reaction_times), 2)
    accuracy = round((correct / 3) * 100, 2)

    speak(f"Your average reaction time is {avg_rt} milliseconds.")
    speak(f"Your accuracy is {accuracy} percent.")

    return avg_rt, accuracy


# -------------------------------------------------
# NEW: FINGER TAPPING TEST
# -------------------------------------------------
def finger_tapping_test():
    speak("Finger tapping test selected.")
    speak("You will first do one practice trial.")
    speak("Then three trials will be recorded.")
    speak("Tap the enter key as fast as possible when prompted.")

    # -------- PRACTICE --------
    print("\nPRACTICE TRIAL (Not recorded)")
    speak("Practice trial. Start tapping now.")
    start = time.time()
    while time.time() - start < 3:
        input()
    speak("Practice completed.")

    # -------- RECORDED --------
    tap_counts = []

    for i in range(3):
        speak(f"Trial number {i + 1}. Start tapping.")
        print("Tap ENTER repeatedly for 5 seconds.")

        count = 0
        start = time.time()
        while time.time() - start < 5:
            input()
            count += 1

        tap_counts.append(count)
        speak(f"You tapped {count} times.")

    avg_taps = round(sum(tap_counts) / len(tap_counts), 2)
    speak(f"Your average tapping count is {avg_taps} taps.")
    return avg_taps


# ---------------- MAIN MENU ----------------
def main():
    speak("Reaction time module started.")

    while True:
        speak("Please choose a test.")

        print("\nChoose a test:")
        print("1. Simple Visual Reaction Test")
        print("2. Choice Reaction Test")
        print("3. Finger Tapping Test")
        print("4. Exit")

        speak("Press one for simple reaction. Press two for choice reaction. Press three for finger tapping. Press four to exit.")

        choice = input("Enter your choice: ").strip()
        speak(f"You selected option {choice}.")

        if choice == "1":
            simple_visual_reaction()
        elif choice == "2":
            choice_reaction_test()
        elif choice == "3":
            finger_tapping_test()
        elif choice == "4":
            speak("Exiting reaction time module. Thank you.")
            break
        else:
            speak("Invalid choice. Please try again.")


if __name__ == "__main__":
    main()
