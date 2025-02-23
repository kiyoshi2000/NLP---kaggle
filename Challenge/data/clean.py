import pandas as pd

def clean_csv(input_file, output_file):

    df = pd.read_csv(input_file) 
    df = df.iloc[:, [0, -1]]
    df.to_csv(output_file, index=False)
    
# Example usage
clean_csv('submission.csv', 'cleaned_submission.csv')
