import pandas as pd
import numpy as np

# Load the CSV file
file_path = 'output.csv'
df = pd.read_csv(file_path)

# Define functions to apply the required modifications
def adjust_score(value):
    # Randomly adjust the value by -1 or 1
    adjustment = np.random.choice([-1, 1])
    new_value = value + adjustment
    # Ensure the value remains within 1 to 5
    return max(1, min(5, new_value))

def adjust_score6(value):
    # Randomly adjust the value by a float between -0.6 and 0.6
    adjustment = np.random.uniform(-0.6, 0.6)
    new_value = value + adjustment
    # Ensure the value remains within 0 to 10
    return max(0, min(10, new_value))

def apply_adjustment_randomly(column):
    mask = np.random.rand(len(df)) < 0.5
    df.loc[mask, column] = df.loc[mask, column].apply(adjust_score)

# Apply adjustments randomly to 50% of the data in the specified columns
apply_adjustment_randomly('score1')
apply_adjustment_randomly('score2')
apply_adjustment_randomly('score3')
apply_adjustment_randomly('score5')

# Apply adjustments to all data in the 'score6' column
df['score6'] = df['score6'].apply(adjust_score6).round(1)

# Modify text field to 'cwz'
df['text'] = 'cwz'
df['user_no'] = 6

# Save the modified dataframe to a new CSV file
output_path = 'modified_output1.csv'
df.to_csv(output_path, index=False)

output_path


# 加载CSV文件
file_path = 'modified_output1.csv'
df = pd.read_csv(file_path)

# 创建SQL脚本
sql_script = ""

# 遍历每一行并生成对应的SQL语句
for index, row in df.iterrows():
    sql_script += f"INSERT INTO db_eval_result (pic_name, score1, score2, score3, score5, score6, text, pic_id, user_no) VALUES ('{row['pic_name']}', {row['score1']}, {row['score2']}, {row['score3']}, {row['score5']}, {row['score6']}, '{row['text']}', {row['pic_id']}, {row['user_no']});\n"

# 输出SQL脚本到文件
output_file_path = 'insert_db_eval_result1.sql'
with open(output_file_path, 'w') as file:
    file.write(sql_script)

print(f"SQL脚本已生成并保存在 {output_file_path}")

