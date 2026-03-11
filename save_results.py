import csv
from datetime import datetime
import os

# Save reaction test results
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


# Save memory test results
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
                "stroop_reaction_time",
                "stroop_accuracy"
            ])

        writer.writerow([
            datetime.now(),
            word,
            number,
            stroop_rt,
            stroop_acc
        ])