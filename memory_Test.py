import time
import random
import pyttsx3

# -------------------------------------------------
# SAFE TEXT-TO-SPEECH (reinitialized every call)
# -------------------------------------------------
def speak(text):
    print("[VOICE]:", text)
    engine = pyttsx3.init()
    engine.setProperty('rate', 135)
    engine.setProperty('volume', 1.0)
    engine.say(text)
    engine.runAndWait()
    engine.stop()
    time.sleep(0.6)


def clear_screen():
    print("\n" * 50)


# -------------------------------------------------
# 1. WORD RECALL TEST
# -------------------------------------------------
def word_recall_test():
    speak("Word recall test selected.")
    speak("You will first do one practice trial, followed by three recorded trials.")

    word_bank = ["apple", "chair", "river", "house", "book", "green", "clock", "phone"]

    # ---------- PRACTICE ----------
    speak("Practice trial. Memorize the words.")
    practice_words = random.sample(word_bank, 4)
    print("Words:", " ".join(practice_words))

    time.sleep(5)
    clear_screen()
    input("Enter recalled words: ")

    speak("Practice completed. Recorded trials will start now.")

    # ---------- RECORDED ----------
    scores = []

    for i in range(3):
        speak(f"Trial {i + 1}. Memorize the words.")
        words = random.sample(word_bank, 5)
        print("Words:", " ".join(words))

        time.sleep(6)
        clear_screen()

        recall = input("Enter recalled words: ").lower().split()
        correct = len(set(recall) & set(words))
        scores.append(correct)

        speak(f"You recalled {correct} words correctly.")

    avg_score = round(sum(scores) / len(scores), 2)
    speak(f"Your average word recall score is {avg_score} out of five.")

    return avg_score


# -------------------------------------------------
# 2. NUMBER SEQUENCE RECALL
# -------------------------------------------------
def number_sequence_test():
    speak("Number sequence recall test selected.")
    speak("You will first do one practice trial, followed by three recorded trials.")

    # ---------- PRACTICE ----------
    speak("Practice trial. Memorize the numbers.")
    practice_seq = [random.randint(0, 9) for _ in range(4)]
    print("Sequence:", " ".join(map(str, practice_seq)))

    time.sleep(4)
    clear_screen()
    input("Enter sequence: ")

    speak("Practice completed. Recorded trials will start now.")

    # ---------- RECORDED ----------
    scores = []

    for i in range(3):
        speak(f"Trial {i + 1}. Memorize the numbers.")
        sequence = [random.randint(0, 9) for _ in range(6)]
        print("Sequence:", " ".join(map(str, sequence)))

        time.sleep(5)
        clear_screen()

        response = input("Enter sequence: ").split()
        correct = sum(
            1 for j in range(min(len(sequence), len(response)))
            if response[j] == str(sequence[j])
        )

        scores.append(correct)
        speak(f"You recalled {correct} numbers correctly.")

    avg_score = round(sum(scores) / len(scores), 2)
    speak(f"Your average number recall score is {avg_score} out of six.")

    return avg_score


# -------------------------------------------------
# 3. STROOP TEST
# -------------------------------------------------
def stroop_test():
    speak("Stroop test selected.")
    speak("Type the color of the word, not the word itself.")
    speak("You will first do one practice trial, followed by three recorded trials.")

    colors = ["RED", "GREEN", "BLUE", "YELLOW"]

    # ---------- PRACTICE ----------
    speak("Practice trial.")
    word = random.choice(colors)
    color = random.choice(colors)
    print(f"Word: {word} (Color shown: {color})")

    time.sleep(3)
    clear_screen()
    input("Enter the COLOR: ")

    speak("Practice completed. Recorded trials will start now.")

    # ---------- RECORDED ----------
    correct_scores = []
    reaction_times = []

    for i in range(3):
        speak(f"Trial {i + 1}.")
        word = random.choice(colors)
        color = random.choice(colors)
        print(f"Word: {word} (Color shown: {color})")

        time.sleep(2)
        clear_screen()

        start = time.time()
        response = input("Enter the COLOR: ").strip().upper()
        end = time.time()

        rt = round((end - start) * 1000, 2)
        reaction_times.append(rt)

        correct = 1 if response == color else 0
        correct_scores.append(correct)

        speak(f"Reaction time {rt} milliseconds.")

    avg_rt = round(sum(reaction_times) / len(reaction_times), 2)
    accuracy = round((sum(correct_scores) / 3) * 100, 2)

    speak(f"Your average response time is {avg_rt} milliseconds.")
    speak(f"Your accuracy is {accuracy} percent.")

    return avg_rt, accuracy


# -------------------------------------------------
# MAIN MENU (VOICE + TEXT)
# -------------------------------------------------
def main():
    speak("Memory test module started.")

    while True:
        speak("Please choose a memory test.")
        speak("Press one for word recall.")
        speak("Press two for number sequence recall.")
        speak("Press three for Stroop test.")
        speak("Press four to exit.")

        print("\nChoose a test:")
        print("1. Word Recall Test")
        print("2. Number Sequence Recall")
        print("3. Stroop Test")
        print("4. Exit")

        choice = input("Enter your choice: ").strip()
        speak(f"You selected option {choice}.")

        if choice == "1":
            word_recall_test()
        elif choice == "2":
            number_sequence_test()
        elif choice == "3":
            stroop_test()
        elif choice == "4":
            speak("Exiting memory test module. Thank you.")
            break
        else:
            speak("Invalid choice. Please try again.")


if __name__ == "__main__":
    main()
