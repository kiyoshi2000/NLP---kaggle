import json
import os 

data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
submission_path = os.path.join(data_dir, 'submission.csv')
y_pred_test = os.path.join(data_dir, 'y_pred_test.txt')


'''
with open('y_pred_shuffle.txt', 'r') as file:
    datas = file.readlines()

with open('y_test_shuffle_for_kaggle.txt', 'w') as file:
    i = 0
    file.write(f"ID,Usage,Label\n")
    for data in datas:
        file.write(f"{i},Private,{data}") if i % 2 == 1 else file.write(f"{i},Public,{data}")
        i += 1

with open('y_test_shuffle.txt', 'r') as file:
    datas = file.readlines()

with open('y_test_shuffle_for_kaggle.txt', 'w') as file:
    i = 0
    file.write(f"ID,Usage,Label\n")
    for data in datas:
        file.write(f"{i},Private,{data}")
        i += 1

'''

# 1. Read test set predictions
with open(y_pred_test, 'r', encoding='utf-8') as file:
    predictions = file.readlines()

# 2. Create submission file
with open(submission_path, 'w', encoding='utf-8') as file:
    file.write("ID,Usage,Label\n")
    for i, pred in enumerate(predictions):
        usage = "Public" if i % 2 == 0 else "Private"
        file.write(f"{i},{usage},{pred.strip()}\n")
