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

cgpa_mapping = {1: '<2.00', 2: '2.00-2.49', 3: '2.50-2.99', 4: '3.00-3.49', 5: 'above 3.49'}
df['CUML_GPA'] = df['CUML_GPA'].map(cgpa_mapping)

exgpa_mapping = {1: '<2.00', 2: '2.00-2.49', 3: '2.50-2.99', 4: '3.00-3.49', 5: 'above 3.49'}
df['EXP_GPA'] = df['EXP_GPA'].map(exgpa_mapping)

df.to_csv('updated_student_prediction2.csv', index=False)

print(df)