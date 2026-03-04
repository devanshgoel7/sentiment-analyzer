from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments
from transformers import Trainer
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

dataset = load_dataset("imdb")

# tokenizing words to numbers
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

small_train = dataset["train"].shuffle(seed=42).select(range(2000))
small_test = dataset["test"].shuffle(seed=42).select(range(500))

def tokenize(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=256)

# dividing test and train 
small_train = small_train.map(tokenize,batched=True)
small_test = small_test.map(tokenize,batched=True)

small_train.set_format("torch", columns=["input_ids", "attention_mask", "label"])
small_test.set_format("torch", columns=["input_ids", "attention_mask", "label"])

# model

model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

# training

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    learning_rate=3e-5,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True
)

# checking accuracy of the model
def compute_metrics(eval_pred):
    logits, labels = eval_pred  
    predictions = np.argmax(logits, axis=-1) 
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions)
    }

trainer = Trainer(
    model = model,
    args = training_args,
    train_dataset= small_train,
    eval_dataset= small_test,
    compute_metrics= compute_metrics
)

trainer.train()
