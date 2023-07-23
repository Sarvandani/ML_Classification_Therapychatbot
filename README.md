```python
import pandas as pd
import re
import nltk
import torch
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, AdamW

```


```python
data = pd.read_csv("./Sheet_1.csv",encoding= "latin1" )
data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>response_id</th>
      <th>class</th>
      <th>response_text</th>
      <th>Unnamed: 3</th>
      <th>Unnamed: 4</th>
      <th>Unnamed: 5</th>
      <th>Unnamed: 6</th>
      <th>Unnamed: 7</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>response_1</td>
      <td>not_flagged</td>
      <td>I try and avoid this sort of conflict</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>response_2</td>
      <td>flagged</td>
      <td>Had a friend open up to me about his mental ad...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>response_3</td>
      <td>flagged</td>
      <td>I saved a girl from suicide once. She was goin...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>response_4</td>
      <td>not_flagged</td>
      <td>i cant think of one really...i think i may hav...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>response_5</td>
      <td>not_flagged</td>
      <td>Only really one friend who doesn't fit into th...</td>
      <td></td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
data.info()

```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 80 entries, 0 to 79
    Data columns (total 8 columns):
     #   Column         Non-Null Count  Dtype  
    ---  ------         --------------  -----  
     0   response_id    80 non-null     object 
     1   class          80 non-null     object 
     2   response_text  80 non-null     object 
     3   Unnamed: 3     2 non-null      object 
     4   Unnamed: 4     0 non-null      float64
     5   Unnamed: 5     1 non-null      object 
     6   Unnamed: 6     0 non-null      float64
     7   Unnamed: 7     1 non-null      object 
    dtypes: float64(2), object(6)
    memory usage: 5.1+ KB



```python
#checking missing values
null_values = data.isnull().sum()
print(null_values)
```

    response_id       0
    class             0
    response_text     0
    Unnamed: 3       78
    Unnamed: 4       80
    Unnamed: 5       79
    Unnamed: 6       80
    Unnamed: 7       79
    dtype: int64



```python
#checking duplicated values
duplicate_values = data[data.duplicated()]
print(duplicate_values)
```

    Empty DataFrame
    Columns: [response_id, class, response_text, Unnamed: 3, Unnamed: 4, Unnamed: 5, Unnamed: 6, Unnamed: 7]
    Index: []



```python
# feature engineering :  
#Dropping some of the redundant features
to_drop = ["Unnamed: 3", "Unnamed: 4","Unnamed: 5", "Unnamed: 6", "Unnamed: 7"]
data = data.drop(to_drop, axis=1)
data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>response_id</th>
      <th>class</th>
      <th>response_text</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>response_1</td>
      <td>not_flagged</td>
      <td>I try and avoid this sort of conflict</td>
    </tr>
    <tr>
      <th>1</th>
      <td>response_2</td>
      <td>flagged</td>
      <td>Had a friend open up to me about his mental ad...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>response_3</td>
      <td>flagged</td>
      <td>I saved a girl from suicide once. She was goin...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>response_4</td>
      <td>not_flagged</td>
      <td>i cant think of one really...i think i may hav...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>response_5</td>
      <td>not_flagged</td>
      <td>Only really one friend who doesn't fit into th...</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Download stopwords and punkt tokenizer from NLTK
nltk.download('stopwords')
nltk.download('punkt')
```

    [nltk_data] Downloading package stopwords to
    [nltk_data]     /Users/mohamadians/nltk_data...
    [nltk_data]   Package stopwords is already up-to-date!
    [nltk_data] Downloading package punkt to
    [nltk_data]     /Users/mohamadians/nltk_data...
    [nltk_data]   Package punkt is already up-to-date!





    True




```python
# clean the text data
def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)  # Remove special characters
    word_tokens = word_tokenize(text)  # Tokenize the text
    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in word_tokens if word not in stop_words]  # Remove stopwords
    return ' '.join(filtered_words)
```


```python
# Apply preprocessing to the 'response_text' column
df = pd.DataFrame(data)
df['processed_text'] = df['response_text'].apply(preprocess_text)

```


```python
# Split the data into training and testing 
X_train, X_test, y_train, y_test = train_test_split(
    df['processed_text'], df['class'], test_size=0.3, random_state=42)
```


```python
# Load the DistilBERT tokenizer and model
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')
```

    Some weights of DistilBertForSequenceClassification were not initialized from the model checkpoint at distilbert-base-uncased and are newly initialized: ['classifier.weight', 'pre_classifier.bias', 'classifier.bias', 'pre_classifier.weight']
    You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.



```python
# Tokenize the input text and convert to tensors
X_train_tokens = tokenizer(X_train.tolist(), padding=True, truncation=True, return_tensors='pt', max_length=128)
X_test_tokens = tokenizer(X_test.tolist(), padding=True, truncation=True, return_tensors='pt', max_length=128)

```


```python
# Convert the class labels to numerical values
class_mapping = {"not_flagged": 0, "flagged": 1}
y_train_numerical = torch.tensor([class_mapping[label] for label in y_train.tolist()])
y_test_numerical = torch.tensor([class_mapping[label] for label in y_test.tolist()])
```


```python
# Create the DataLoader
train_data = torch.utils.data.TensorDataset(X_train_tokens['input_ids'], X_train_tokens['attention_mask'], y_train_numerical)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=16, shuffle=True)
```


```python
# Optimizer and loss function
optimizer = AdamW(model.parameters(), lr=2e-5)
loss_fn = torch.nn.CrossEntropyLoss()
```

    /Users/mohamadians/opt/anaconda3/lib/python3.8/site-packages/transformers/optimization.py:411: FutureWarning: This implementation of AdamW is deprecated and will be removed in a future version. Use the PyTorch implementation torch.optim.AdamW instead, or set `no_deprecation_warning=True` to disable this warning
      warnings.warn(



```python
# Training loop
epochs = 15
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
for epoch in range(epochs):
    model.train()
    for batch in train_loader:
        input_ids, attention_mask, targets = batch
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        targets = targets.to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=targets)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
```


```python
# Evaluation
model.eval()
with torch.no_grad():
    X_test_tokens = {k: v.to(device) for k, v in X_test_tokens.items()}
    outputs = model(**X_test_tokens)
    logits = outputs.logits
    _, predicted_labels = torch.max(logits, 1)

y_pred = predicted_labels.cpu().numpy()
y_test = y_test_numerical.cpu().numpy()

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=class_mapping.keys()))
```

    Accuracy: 0.7083333333333334
    
    Classification Report:
                   precision    recall  f1-score   support
    
     not_flagged       0.87      0.72      0.79        18
         flagged       0.44      0.67      0.53         6
    
        accuracy                           0.71        24
       macro avg       0.66      0.69      0.66        24
    weighted avg       0.76      0.71      0.72        24
    



```python

```


```python

```
