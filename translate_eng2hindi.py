# English to Hindi Translation Model with Attention
# Includes training, evaluation, and attention plotting

import tensorflow as tf
import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.models import Model

# Preprocessing functions
def preprocess_english(w):
    w = w.lower().strip()
    w = re.sub(r"([?.!,¿])", r" \1 ", w)
    w = re.sub(r"[^a-zA-Z?.!,¿]+", " ", w)
    return '<start> ' + w.strip() + ' <end>'

def preprocess_hindi(w):
    w = str(w).strip()
    return '<start> ' + w + ' <end>'
"""
# Load dataset with utf-8 and drop NaNs
df = pd.read_csv('./data/english-hindi-translation.csv', encoding='utf-8')
df = df.dropna(subset=['English', 'Hindi'])
df = df.iloc[:10000]  # Limit for quick training
"""

# Download via Kaggle API if available, then load:
df = pd.read_csv('/Users/pallavipriya/Documents/AI Projects/Language_Translatio_Model/Dataset_English_Hindi.csv', encoding='utf-8')
df_small = df.dropna(subset=['English', 'Hindi'])  # Drop rows where either column is NaN
# Limit to first 10,000 rows
#df = df.iloc[:3000]
df = df_small.iloc[:200]
print("Sample data:")
print(df.head())
print(df.columns)  # Shows column names


# Preprocess
input_texts = df['English'].apply(preprocess_english).tolist()
target_texts = df['Hindi'].apply(preprocess_hindi).tolist()

# Tokenize
input_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
target_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
input_tokenizer.fit_on_texts(input_texts)
target_tokenizer.fit_on_texts(target_texts)

input_sequences = input_tokenizer.texts_to_sequences(input_texts)
target_sequences = target_tokenizer.texts_to_sequences(target_texts)

# Pad sequences
max_input_len = max(len(seq) for seq in input_sequences)
max_target_len = max(len(seq) for seq in target_sequences)

input_tensor = tf.keras.preprocessing.sequence.pad_sequences(input_sequences, maxlen=max_input_len, padding='post')
target_tensor = tf.keras.preprocessing.sequence.pad_sequences(target_sequences, maxlen=max_target_len, padding='post')

# Vocab sizes
input_vocab_size = len(input_tokenizer.word_index) + 1
target_vocab_size = len(target_tokenizer.word_index) + 1

# Train-test split
input_train, input_val, target_train, target_val = train_test_split(input_tensor, target_tensor, test_size=0.2)

# Encoder
def Encoder(vocab_size, embedding_dim, units):
    inputs = tf.keras.Input(shape=(None,))
    x = Embedding(vocab_size, embedding_dim)(inputs)
    output, state_h, state_c = LSTM(units, return_sequences=True, return_state=True)(x)
    return tf.keras.Model(inputs, [output, state_h, state_c])

# Bahdanau Attention
class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        super().__init__()
        self.W1 = Dense(units)
        self.W2 = Dense(units)
        self.V = Dense(1)

    def call(self, query, values):
        query = tf.expand_dims(query, 1)
        score = self.V(tf.nn.tanh(self.W1(values) + self.W2(query)))
        attention_weights = tf.nn.softmax(score, axis=1)
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector, attention_weights

# Decoder
class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, dec_units):
        super().__init__()
        self.embedding = Embedding(vocab_size, embedding_dim)
        self.lstm = LSTM(dec_units, return_sequences=True, return_state=True)
        self.fc = Dense(vocab_size)
        self.attention = BahdanauAttention(dec_units)

    def call(self, x, enc_output, state_h):
        context_vector, attention_weights = self.attention(state_h, enc_output)
        x = self.embedding(x)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
        output, h, c = self.lstm(x)
        output = self.fc(output)
        return output, h, c, attention_weights

# Initialize
embedding_dim = 64
units = 128
encoder = Encoder(input_vocab_size, embedding_dim, units)
decoder = Decoder(target_vocab_size, embedding_dim, units)

optimizer = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    return tf.reduce_mean(loss_)

# Training loop
epochs = 500
batch_size = 2

for epoch in range(epochs):
    total_loss = 0
    for i in range(0, len(input_train), batch_size):
        inp = input_train[i:i+batch_size]
        targ = target_train[i:i+batch_size]

        with tf.GradientTape() as tape:
            enc_output, enc_hidden, enc_cell = encoder(inp)
            dec_input = tf.expand_dims([target_tokenizer.word_index['<start>']] * len(inp), 1)
            dec_hidden = enc_hidden
            loss = 0

            for t in range(1, targ.shape[1]):
                predictions, dec_hidden, _, _ = decoder(dec_input, enc_output, dec_hidden)
                loss += loss_function(targ[:, t], predictions[:, 0, :])
                dec_input = tf.expand_dims(targ[:, t], 1)

        batch_loss = loss / targ.shape[1]
        total_loss += batch_loss

        variables = encoder.trainable_variables + decoder.trainable_variables
        gradients = tape.gradient(loss, variables)
        optimizer.apply_gradients(zip(gradients, variables))

    print(f"Epoch {epoch+1}, Loss {total_loss.numpy():.4f}")

# Evaluation
def evaluate(sentence):
    sentence = preprocess_english(sentence)
    inputs = input_tokenizer.texts_to_sequences([sentence])
    inputs = tf.keras.preprocessing.sequence.pad_sequences(inputs, maxlen=max_input_len, padding='post')
    enc_out, enc_hidden, enc_cell = encoder(inputs)
    dec_hidden = enc_hidden
    dec_input = tf.expand_dims([target_tokenizer.word_index['<start>']], 0)

    result = ''
    attentions = []

    for t in range(max_target_len):
        predictions, dec_hidden, _, attention_weights = decoder(dec_input, enc_out, dec_hidden)
        predicted_id = tf.argmax(predictions[0][0]).numpy()
        predicted_word = target_tokenizer.index_word.get(predicted_id, '')
        attentions.append(tf.squeeze(attention_weights, axis=-1)[0].numpy())
        if predicted_word == '<end>' or predicted_word == '':
            break
        result += predicted_word + ' '
        dec_input = tf.expand_dims([predicted_id], 0)

    return result.strip(), attentions

def translate(sentence):
    result, attention = evaluate(sentence)
    print(f"\n🗣️ Input: {sentence}")
    print(f"📝 Translation: {result}")
    return result
"""
def plot_attention(attention, input_sentence, translation):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(1, 1, 1)
    attention = np.array(attention)
    cax = ax.matshow(attention, cmap='viridis')
    fig.colorbar(cax)
    ax.set_xticks(range(len(input_sentence.split())))
    ax.set_xticklabels(input_sentence.split(), rotation=45)
    ax.set_yticks(range(len(translation.split())))
    ax.set_yticklabels(translation.split())
    ax.set_xlabel('Input')
    ax.set_ylabel('Output')
    plt.show()

"""
# Try translating a few sentences
sentences_to_translate = ["hello", "thank you", "good night", "i love you"]
for sentence in sentences_to_translate:
    translation, attention = evaluate(sentence)
    print(f"\nTranslate: {sentence} ➡️ {translation}")

"""
# Test example
sentence = "i love you"
translation, attention = evaluate(sentence)
print(f"\nTranslate: {sentence} ➡️ {translation}")
#plot_attention(attention, preprocess_english(sentence).replace('<start>', '').replace('<end>', '').strip(), translation)

"""