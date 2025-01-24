# sentiment_analysis
I this repo we will perfrom sentiment analysis using IMDB dataset to fine-tune a DistilBERT model for sentiment analysis. We will follow this link for this project:

https://huggingface.co/blog/sentiment-analysis-python

# Activate GPU and Install Dependencies

As a first step, let's set up Google Colab to use a GPU (instead of CPU) to train the model much faster. You can do this by going to the menu, clicking on 'Runtime' > 'Change runtime type', and selecting 'GPU' as the Hardware accelerator. Once you do this, you should check if GPU is available on our notebook by running the following code:

`import torch`
`torch.cuda.is_available()`

Then, install the libraries you will be using in this tutorial:

`!pip install datasets transformers huggingface_hub`

You should also install git-lfs to use git in our model repository:

`!apt-get install git-lfs`

# Preprocess data

You need data to fine-tune DistilBERT for sentiment analysis. So, let's use 🤗Datasets library to download and preprocess the IMDB dataset so you can then use this data for training your model:

`from datasets import load_dataset`
`imdb = load_dataset("imdb")`

IMDB is a huge dataset, so let's create smaller datasets to enable faster training and testing:

`small_train_dataset = imdb["train"].shuffle(seed=42).select([i for i in list(range(3000))])`
`small_test_dataset = imdb["test"].shuffle(seed=42).select([i for i in list(range(300))])`

To preprocess our data, you will use DistilBERT tokenizer:

`from transformers import AutoTokenizer`
`tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")`

Next, you will prepare the text inputs for the model for both splits of our dataset (training and test) by using the map method:

`def preprocess_function(examples):`
   `return tokenizer(examples["text"], truncation=True)`
 
`tokenized_train = small_train_dataset.map(preprocess_function, batched=True)`
`tokenized_test = small_test_dataset.map(preprocess_function, batched=True)`

To speed up training, let's use a data_collator to convert your training samples to PyTorch tensors and concatenate them with the correct amount of padding:

`from transformers import DataCollatorWithPadding`
`data_collator = DataCollatorWithPadding(tokenizer=tokenizer)`

# Training the model

Now that the preprocessing is done, you can go ahead and train your model 🚀

You will be throwing away the pretraining head of the DistilBERT model and replacing it with a classification head fine-tuned for sentiment analysis. This enables you to transfer the knowledge from DistilBERT to your custom model 🔥

For training, you will be using the Trainer API, which is optimized for fine-tuning Transformers🤗 models such as DistilBERT, BERT and RoBERTa.

First, let's define DistilBERT as your base model:

`from transformers import AutoModelForSequenceClassification`
`model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)`

Then, let's define the metrics you will be using to evaluate how good is your fine-tuned model (accuracy and f1 score):

`pip install --upgrade datasets`

`!pip install evaluate`

`import numpy as np`
`from evaluate import load`

`def compute_metrics(eval_pred):`
   ` # Load evaluation metrics`
    `accuracy_metric = load("accuracy")`
    `f1_metric = load("f1")`

   ` # Unpack predictions and labels`
   ` logits, labels = eval_pred`
   ` predictions = np.argmax(logits, axis=-1)`

   ` # Compute accuracy and F1 score`
    `accuracy = accuracy_metric.compute(predictions=predictions, references=labels)["accuracy"]`
    `f1 = f1_metric.compute(predictions=predictions, references=labels)["f1"]`

    `return {"accuracy": accuracy, "f1": f1}`


Next, let's login to your Hugging Face account so you can manage your model repositories. notebook_login will launch a widget in your notebook where you'll need to add your Hugging Face token:



from huggingface_hub import notebook_login
notebook_login()

You are almost there! Before training our model, you need to define the training arguments and define a Trainer with all the objects you constructed up to this point: 

`import pandas as pd`
`from datasets import Dataset`

Load data from CSV
`df = pd.read_csv("your_dataset.csv")`

Convert to Hugging Face dataset
`train_dataset = Dataset.from_pandas(df)`


from transformers import TrainingArguments, Trainer
 
repo_name = "finetuning-sentiment-model-3000-samples"
 
training_args = TrainingArguments(
   output_dir=repo_name,
   learning_rate=2e-5,
   per_device_train_batch_size=16,
   per_device_eval_batch_size=16,
   num_train_epochs=2,
   weight_decay=0.01,
   save_strategy="epoch",
   push_to_hub=True,
)
 
trainer = Trainer(
   model=model,
   args=training_args,
   train_dataset=tokenized_train,
   eval_dataset=tokenized_test,
   tokenizer=tokenizer,
   data_collator=data_collator,
   compute_metrics=compute_metrics,
)

Now, it's time to fine-tune the model on the sentiment analysis dataset! 🙌 You just have to call the train() method of your Trainer:

trainer.train()

And voila! You fine-tuned a DistilBERT model for sentiment analysis! 🎉
