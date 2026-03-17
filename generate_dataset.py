import random

# =========================================================
# DATA SOURCES (high quality, diverse)
# =========================================================

paragraphs = [

# --- Science ---
"The Earth revolves around the Sun, creating day and night cycles.",
"Gravity is a force that attracts objects toward each other.",
"Energy exists in many forms such as heat, light, and motion.",

# --- Technology ---
"Computers process information using binary signals.",
"Programming involves writing instructions step by step.",
"Algorithms are structured methods used to solve problems.",

# --- Stories ---
"Once upon a time there was a curious child who asked many questions.",
"The traveler walked through a quiet forest and listened to the wind.",
"A small invention changed the life of a village.",

# --- Philosophy ---
"Knowledge begins with curiosity and grows through exploration.",
"People learn by observing patterns and asking questions.",
"Clear thinking helps solve complex problems.",

# --- Education ---
"Learning happens through practice and repetition.",
"Teachers explain ideas so others can understand them easily.",
"Reading improves thinking and imagination."
]

# =========================================================
# DIALOGUE DATA
# =========================================================

dialogues = [
"Teacher: What is gravity?\nStudent: It is the force that pulls objects toward Earth.",
"Student: How do computers work?\nTeacher: They follow instructions called programs.",
"Child: Why is the sky blue?\nParent: Because of how light scatters in the atmosphere.",
"Scientist: What do you observe?\nAssistant: The object accelerates over time.",
"Engineer: How can we fix this problem?\nTechnician: We can test each part step by step."
]

# =========================================================
# QA DATA
# =========================================================

qa_pairs = [
"Q: What is energy?\nA: Energy is the ability to do work.",
"Q: What is an algorithm?\nA: It is a step by step method to solve a problem.",
"Q: Why do we learn?\nA: Learning helps us understand and improve.",
"Q: What is programming?\nA: Programming is writing instructions for computers.",
"Q: What is science?\nA: Science is the study of the natural world."
]

# =========================================================
# SENTENCE VARIATIONS
# =========================================================

connectors = [
    "In addition,",
    "Another important idea is that",
    "For example,",
    "As a result,",
    "Over time,",
    "In many cases,"
]

# =========================================================
# DATASET GENERATION
# =========================================================

def generate_dataset(target_chars=100000):
    text = ""

    while len(text) < target_chars:
        mode = random.choice(["paragraph", "dialogue", "qa"])

        if mode == "paragraph":
            p = random.choice(paragraphs)
            c = random.choice(connectors)
            new_para = p + " " + c + " " + p.lower()

        elif mode == "dialogue":
            new_para = random.choice(dialogues)

        else:
            new_para = random.choice(qa_pairs)

        # add spacing for better structure learning
        text += new_para + "\n\n"

        # occasional extra newline (improves formatting learning)
        if random.random() < 0.3:
            text += "\n"

    return text[:target_chars]


# =========================================================
# SAVE DATASET
# =========================================================

dataset = generate_dataset()

with open("data/dataset.txt", "w", encoding="utf-8") as f:
    f.write(dataset)

print(f"Dataset generated with {len(dataset)} characters.")