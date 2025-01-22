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
