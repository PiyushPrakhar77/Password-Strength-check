# Password Strength Classification

A machine learning project that classifies passwords into **Weak**, **Medium**, or **Strong** categories using TF-IDF character features and a Random Forest classifier.

## Features

- Loads a password-strength dataset from a CSV file.
- Removes rows with missing values.
- Maps numeric strength labels (`0`, `1`, `2`) to `Weak`, `Medium`, and `Strong`.
- Converts passwords into character-level TF-IDF features.
- Splits the dataset into training and test sets.
- Trains a `RandomForestClassifier`.
- Evaluates the model using test accuracy.
- Accepts a password interactively and predicts its strength.

## Project structure

```text
password-strength-classifier/
├── password_strength.py
├── requirements.txt
├── README.md
└── data.csv
```

Place your dataset named `data.csv` in the project directory, or update the file path in the Python script.

## Requirements

- Python 3.10 or a compatible Python version
- A CSV dataset containing `password` and `strength` columns

## Installation

### 1. Create and activate a virtual environment

**Windows (Command Prompt):**

```bat
py -3.10 -m venv .venv
.venv\Scripts\activate
```

**macOS/Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Dataset format

The CSV file should contain at least these columns:

| password | strength |
|---|---:|
| example123 | 1 |
| abc | 0 |
| A-long-example! | 2 |

In the dataset used by the original code, the expected numeric labels are `0`, `1`, and `2`. Confirm the label meanings in your actual dataset before mapping them.

**Privacy warning:** Password datasets can contain real or reused credentials. Use a trusted dataset, do not publish raw passwords, and do not collect or store users' real passwords for this experiment.

## Run the project

Save the Python code as `password_strength.py`, put `data.csv` in the same directory, then run:

```bash
python password_strength.py
```

Update the original Windows-specific path, for example:

```python
data = pd.read_csv("data.csv", on_bad_lines="skip")
```

The original script trains the model each time it runs, which can take time for a dataset with hundreds of thousands of rows.

## How it works

1. **Data cleaning:** Missing rows are removed.
2. **Label mapping:** Numeric target values are converted into readable categories.
3. **Character tokenization:** The `word()` function splits each password into individual characters.
4. **TF-IDF:** `TfidfVectorizer` produces a sparse numerical feature matrix from character tokens.
5. **Train/test split:** 95% of the records are used for training and 5% for testing.
6. **Random Forest:** An ensemble of decision trees learns patterns associated with the provided labels.
7. **Evaluation:** Test accuracy measures the proportion of correctly classified test records.
8. **Prediction:** A newly entered password is transformed with the same fitted vectorizer before classification.

## Important implementation notes

- `CountVectorizer` is imported in the original code but not used; it can be removed.
- The original code uses `RandomForestClassifier()` with default settings. Explicitly set `random_state` if you want reproducible model training.
- The reported accuracy of approximately `95.64%` is the result from the supplied run, not a guaranteed result on every environment or dataset.
- Accuracy alone does not show how well each class is predicted. Also evaluate a confusion matrix, precision, recall, and F1-score.
- Character-level TF-IDF features capture character patterns, not whether a password has appeared in a breach or whether it is safe to use.
- The predicted label reflects the training dataset's labels and biases. This model is an educational demonstration, not a complete password-security assessment.

## Security guidance

Do not use this model as the only control for password security. For real applications, use established password-strength guidance and breached-password checks, enforce secure password storage with a modern password-hashing algorithm, and never log passwords. The example uses `getpass` so the entered password is not echoed in a terminal, but it is still processed in memory.

## License

Add a license before distributing this project. Check the dataset's license and usage conditions before sharing it.
