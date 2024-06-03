import pandas as pd
import tiktoken
import json
# FUNCTIONS---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


def num_tokens_from_string(string: str):
    encoding = tiktoken.get_encoding("cl100k_base")
    num_tokens = len(encoding.encode(string))
    return num_tokens


def df_to_format(df):
    formatted_data = []
    for index, row in df.iterrows():
        entry = {"messages":
                 [
                     {'role': 'system', 'content': system_prompt},
                     {'role': 'user', 'content': row["report"]},
                     {'role': 'assistant',
                      'content': row['medical_specialty']},
                 ]}
        formatted_data.append(entry)
    return formatted_data

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


med_reports = pd.read_csv("reports.csv")
# drop all rows where a "report" field is missing/NA.
med_reports = med_reports.dropna(subset=['report'])

# DATASETS: TRAIN -> Validation -> final testing --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# 50 -> 5 -> 5
filtered_reports = med_reports.groupby("medical_specialty")
grouped_data = filtered_reports.sample(60, random_state=42)


valtest_data = grouped_data.groupby(
    "medical_specialty").sample(10, random_state=42)

val = valtest_data.groupby("medical_specialty").head(5)
test = valtest_data.groupby("medical_specialty").tail(5)

# let the train data be the groupedData WITHOUT the valtestdata indices
train_data = grouped_data[~grouped_data.index.isin(valtest_data.index)]


# TOKENIZATION
report_lengths = train_data["report"].apply(num_tokens_from_string)
# print(report_lengths.describe())


# FORMAT DATA
# print(train_data["medical_specialty"].unique())
system_prompt = "Given a medical report, classify it into one of these categories: \n Cardiovascular / Pulmonary, Gastroenterology, Neurology, Radiology,  Surgery"
sample_prompt = {"messages": [
    {'role': 'system', 'content': system_prompt},
    {'role': 'user', 'content': train_data['report'].iloc[0]},
    {'role': 'assistant', 'content': train_data['medical_specialty'].iloc[0]},
]}

# write frmatted training, val data to jsonfile
formatted_training_data = df_to_format(train_data)
with open('fine_tuning_data.jsonl','w') as f:
    for entry in formatted_training_data:
        f.write(json.dumps(entry))
        f.write("\n")

formatted_val_data = df_to_format(val)
with open('val_fine_tuning_data.jsonl','w') as f:
    for entry in formatted_val_data:
        f.write(json.dumps(entry))
        f.write("\n")