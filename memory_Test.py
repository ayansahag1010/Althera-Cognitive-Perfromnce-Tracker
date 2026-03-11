import time
import random
import pyttsx3
import csv
import os
from datetime import datetime


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
    print("\n"*40)


# ---------------- SAVE MEMORY RESULTS ----------------
def save_memory(word=None, number=None, stroop_rt=None, stroop_acc=None):

    filename = "memory_results.csv"
    file_exists = os.path.isfile(filename)

    with open(filename, "a", newline="") as f:

        writer = csv.writer(f)

        if not file_exists:
            writer.writerow([
                "timestamp",
                "word_recall_score",
                "number_recall_score",
                "stroop_reaction_ms",
                "stroop_accuracy_percent"
            ])

        writer.writerow([
            datetime.now(),
            word,
            number,
            stroop_rt,
            stroop_acc
        ])


# ---------------- WORD RECALL ----------------
def word_recall_test():

    word_bank = ["apple","chair","river","house","book","green","clock","phone"]

    speak("Word recall test.")

    practice = random.sample(word_bank,4)
    print("Words:",practice)

    time.sleep(5)
    clear_screen()
    input("Recall words: ")

    scores=[]

    for i in range(3):

        words=random.sample(word_bank,5)
        print("Words:",words)

        time.sleep(6)
        clear_screen()

        recall=input("Recall words: ").lower().split()

        correct=len(set(recall)&set(words))
        scores.append(correct)

        speak(f"You recalled {correct} words.")

    avg_score=round(sum(scores)/len(scores),2)

    save_memory(word=avg_score)

    speak(f"Average word recall {avg_score} out of five.")

    return avg_score


# ---------------- NUMBER RECALL ----------------
def number_sequence_test():

    speak("Number recall test.")

    practice=[random.randint(0,9) for _ in range(4)]
    print("Sequence:",practice)

    time.sleep(4)
    clear_screen()
    input("Enter sequence: ")

    scores=[]

    for i in range(3):

        sequence=[random.randint(0,9) for _ in range(6)]
        print("Sequence:",sequence)

        time.sleep(5)
        clear_screen()

        response=input("Enter sequence: ").split()

        correct=sum(1 for j in range(min(len(sequence),len(response)))
        if response[j]==str(sequence[j]))

        scores.append(correct)

        speak(f"You recalled {correct} numbers.")

    avg_score=round(sum(scores)/len(scores),2)

    save_memory(number=avg_score)

    speak(f"Average number recall {avg_score}.")

    return avg_score


# ---------------- STROOP TEST ----------------
def stroop_test():

    colors=["RED","GREEN","BLUE","YELLOW"]

    speak("Stroop test. Type the color of the word.")

    correct_scores=[]
    reaction_times=[]

    for i in range(3):

        word=random.choice(colors)
        color=random.choice(colors)

        print(f"Word:{word} Color:{color}")

        time.sleep(2)
        clear_screen()

        start=time.time()
        response=input("Enter color: ").upper()
        end=time.time()

        rt=round((end-start)*1000,2)

        reaction_times.append(rt)

        correct=1 if response==color else 0
        correct_scores.append(correct)

        speak(f"Reaction time {rt} milliseconds.")

    avg_rt=round(sum(reaction_times)/3,2)
    accuracy=round((sum(correct_scores)/3)*100,2)

    save_memory(stroop_rt=avg_rt,stroop_acc=accuracy)

    speak(f"Average reaction time {avg_rt} milliseconds.")
    speak(f"Accuracy {accuracy} percent.")

    return avg_rt,accuracy


# ---------------- MENU ----------------
def main():

    speak("Memory test module started.")

    while True:

        print("\n1.Word Recall")
        print("2.Number Recall")
        print("3.Stroop Test")
        print("4.Exit")

        choice=input("Enter choice:")

        if choice=="1":
            word_recall_test()

        elif choice=="2":
            number_sequence_test()

        elif choice=="3":
            stroop_test()

        elif choice=="4":
            break


if __name__=="__main__":
    main()