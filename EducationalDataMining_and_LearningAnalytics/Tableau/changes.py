import pandas as pd

df = pd.read_csv('student_prediction.csv')

notes_mapping = {1: 'Never', 2: 'Sometimes', 3: 'Always'}
df['NOTES'] = df['NOTES'].map(notes_mapping)

attend_mapping = {1: 'Always', 2: 'Sometimes', 3: 'Never'}
df['ATTEND_DEPT'] = df['ATTEND_DEPT'].map(attend_mapping)

grades_mapping = {0: 'Fail', 1: 'DD', 2: 'DC',3: 'CC', 4: 'CB', 5: 'BB',6: 'BA',7: 'AA'}
df['GRADE'] = df['GRADE'].map(grades_mapping)

projects_mapping = {1: 'Positive', 2: 'Negative', 3: 'Neutral'}
df['IMPACT'] = df['IMPACT'].map(projects_mapping)

df.to_csv('updated_student_prediction.csv', index=False)

print(df)
