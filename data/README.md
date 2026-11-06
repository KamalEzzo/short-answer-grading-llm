# Data

The dataset is **not included** in this repository. It is held under restricted access on Zenodo:
[10.5281/zenodo.18856922](https://doi.org/10.5281/zenodo.18856922). Access is granted on request for
non-commercial research, under conditions that forbid any attempt to identify students and any
sharing of individual answers.

After approval, place the file here as `data/Data.xlsx` (this folder's `*.xlsx` files are ignored by git).

## Expected file: `data/Data.xlsx`

One sheet, 2,550 rows (85 questions x 30 answers):

| Column | Content |
|---|---|
| `ID` | answer ID, `Q<question> Student<n>` (for example `Q4 Student1`) |
| `Question` | question text |
| `Model Answer` | the instructor's model answer |
| `Answer Text` | the student's answer (the placeholder `(left blank)` for No Answer) |
| `Clarity_Score`, `Terminology_Score`, `Coverage_Score` | examiner scores, 0-2 |
| `Accuracy_Score` | examiner score, 0-4 |
| `Rubric_Grade` | sum of the four scores divided by two (0-5) |
| `off_topic` | `On-Topic`, `Off-Topic` or `No Answer` |
| `Off_Topic_Reason` | the examiner's reason label |

The data contain no student names, identification numbers or course grades.

## Files in this folder

| File | Content |
|---|---|
| `splits.csv` | question-level split: 51 train, 17 validation, 17 test questions |
| `fewshot_example_ids.csv` | IDs, order and score band of the five few-shot examples (training questions only) |

Files created locally by the scripts and **not** published (they contain student text):
`fewshot_examples.json`, `prompt_example_*.txt`.
