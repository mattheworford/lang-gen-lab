import tensorflow as tf
import numpy as np

from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard

with open("./resources/training_text.txt", "r") as file:
    text = file.read()

# Create character-level vocabulary
vocab = sorted(set(text))
char2idx = {u: i for i, u in enumerate(vocab)}
idx2char = np.array(vocab)

# Convert text to numerical representation
text_as_int = np.array([char2idx[c] for c in text])


# Create training examples and targets
seq_length = 100
batch_size = 64
examples_per_epoch = len(text) // (seq_length + 1)
char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)
sequences = char_dataset.batch(seq_length + 1, drop_remainder=True)


def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text


dataset = sequences.map(split_input_target)

# Define model parameters
vocab_size = len(vocab)
embedding_dim = 256
rnn_units = 1024
epochs = 10


# Define the model architecture
def build_model(vocab_size, embedding_dim, rnn_units):
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Embedding(vocab_size, embedding_dim),
            tf.keras.layers.GRU(
                rnn_units,
                return_sequences=True,
                stateful=True,
                recurrent_initializer="glorot_uniform",
            ),
            tf.keras.layers.Dense(vocab_size),
        ]
    )
    return model


# Build the model
model = build_model(vocab_size, embedding_dim, rnn_units)

# Compile the model
model.compile(
    optimizer="adam", loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True)
)

# Define callbacks
checkpoint_cb = ModelCheckpoint("model.keras", save_best_only=True)
tensorboard_cb = TensorBoard("logs")

# Reshape dataset to have three dimensions
dataset = dataset.batch(batch_size, drop_remainder=True)
dataset = dataset.map(lambda x, y: (tf.reshape(x, [batch_size, seq_length]), y))

train_size = int(0.8 * len(text_as_int))
train_dataset = dataset.take(train_size)
val_dataset = dataset.skip(train_size)

# Train the model
model.fit(
    train_dataset,
    epochs=epochs,
    validation_data=val_dataset,
    callbacks=[checkpoint_cb, tensorboard_cb],
)
