import pandas as pd
import sys
sys.path.insert(0, 'src')
from judge import RAGJudge

data = {
    'Current User Question': ['What is machine learning?']*3,
    'Assistant Answer': ['ML is AI subset', 'ML is specific approach', 'Example: image recognition'],
    'Fragment Texts': ['ML is branch of AI']*3,
    'Conversation History': ['', 'User: What is machine learning?\nAssistant: ML is AI subset', 'User: What is machine learning?\nAssistant: ML is AI subset\nUser: How different?\nAssistant: ML is specific approach']
}
df = pd.DataFrame(data)
print('Input data:')
for i, row in df.iterrows():
    hist = row['Conversation History']
    print(f'Row {i}: hist="{hist}", len={len(hist)}, stripped="{str(hist).strip()}", has_content={bool(str(hist).strip())}')

judge = RAGJudge()
df.to_csv('debug_test.csv', index=False)
result = judge.evaluate_dataset('debug_test.csv')
print(f'\nResults has_conversation_history: {result["has_conversation_history"].tolist()}')
