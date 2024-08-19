import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
import nltk

nltk.download('punkt')

# 데이터 로드
train_df = pd.read_csv('train.csv', header=None)
test_df = pd.read_csv('test.csv', header=None)

# 전처리
# clean = 0, bad = 1
train_texts = []
train_labels = []
test_texts = []
test_labels = []

for _, row in train_df[1:].iterrows():
  if pd.isna(row[0]):
    train_texts.append(row[4])
    train_labels.append(0 if row[14] == 1.0 else 1)
  else:
    train_texts.append(row[0])
    train_labels.append(0 if row[3] == "none" else 1)

for idx, row in test_df[1:].iterrows():
  if pd.isna(row[0]):
    test_texts.append(row[4])
    test_labels.append(0 if row[14] == 1.0 else 1)
  else:
    test_texts.append(row[0])
    test_labels.append(0 if row[3] == "none" else 1)

train_df = pd.DataFrame({
    "text": train_texts,
    "label": train_labels
})
test_df = pd.DataFrame({
    "text": test_texts,
    "label": test_labels
})

train_df.plot(title="Train DataFrame Plot", kind='hist')

# Tokenizer 및 모델 준비
model_name = "beomi/kcbert-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = TFAutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# 문자열 인코딩
def encode(df, tokenizer):
    return tokenizer(df['text'].tolist(), truncation=True, padding=True, return_tensors="tf")

train_encodings = encode(train_df, tokenizer)
test_encodings = encode(test_df, tokenizer)

# 훈련 데이터셋 준비
train_dataset = tf.data.Dataset.from_tensor_slices((
    dict(train_encodings),
    train_df['label'].tolist()
)).shuffle(len(train_df)).batch(8)

test_dataset = tf.data.Dataset.from_tensor_slices((
    dict(test_encodings),
    test_df['label'].tolist()
)).batch(8)

# 모델 컴파일
optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

# 모델 학습
model.fit(train_dataset, validation_data=test_dataset, epochs=3)

# 저장
save_directory = "./saved_model"
model.save_pretrained(save_directory)
tokenizer.save_pretrained(save_directory)

# 불러오기
save_directory = "./saved_model"
loaded_model = TFAutoModelForSequenceClassification.from_pretrained(save_directory)
loaded_tokenizer = AutoTokenizer.from_pretrained(save_directory)

# 평가
optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
loaded_model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

loaded_model.evaluate(test_dataset)

# 마스킹 함수
def mask_profanity(text, model, tokenizer):
    result = []
    for sentence in nltk.sent_tokenize(text):
        tokens = tokenizer.tokenize(sentence)
        token_ids = tokenizer.convert_tokens_to_ids(tokens)
        token_inputs = {'input_ids': tf.expand_dims(token_ids, 0)}
        token_outputs = model(token_inputs)
        sm = tf.nn.softmax(token_outputs.logits, axis=1)

        if sm[0][1] >= 0.7675:
            masked = []
            for word in sentence.split(' '):
                tokens = tokenizer.tokenize(word)
                token_ids = tokenizer.convert_tokens_to_ids(tokens)
                for token_id in token_ids:
                    token_tensor = tf.constant([token_id])
                    token_inputs = {'input_ids': tf.expand_dims(token_tensor, 0)}
                    token_outputs = model(token_inputs)
                    sm = tf.nn.softmax(token_outputs.logits, axis=1)
                    if sm[0][1] >= 0.7677:
                        masked.append('*' * len(tokenizer.decode(token_id).replace('#', '')))
                    else:
                        masked.append(tokenizer.decode(token_id).replace('#', ''))
                masked.append(' ')
            result.append(''.join(masked))
        else:
            result.append(sentence)

    return ''.join(result)

# 예시 사용
input_text = "안녕하세요 야발련아"
masked_text = mask_profanity(input_text, loaded_model, loaded_tokenizer)
print(masked_text)