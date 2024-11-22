# README for Text Summarization Notebook

## Introduction
This notebook demonstrates the implementation of text summarization using a combination of extractive and abstractive methods. It utilizes the CNN/Daily Mail dataset to train and evaluate models for generating summaries of news articles.

### Key Features:
1. **Extractive Summarization**: Identifies key sentences from articles.
2. **Abstractive Summarization**: Generates human-like summaries using transformer models.
3. **Evaluation Metrics**: Measures performance using ROUGE and BLEU scores.
4. **Pretrained Models**: Utilizes pretrained BERT and BART models for summarization.

---

## Requirements
### Libraries and Tools
Ensure the following Python libraries are installed:
- `transformers`
- `datasets`
- `rouge-score`
- `evaluate`
- `nltk`
- `torch`
- `matplotlib`
- `tqdm`

You can install them by running:
```bash
!pip install transformers datasets rouge-score evaluate nltk -q
```

### Dataset
The notebook uses the CNN/Daily Mail dataset, which is automatically downloaded using the `datasets` library.

---

## Notebook Workflow

### Step 1: Dataset Preparation
1. **Loading**: The CNN/Daily Mail dataset is loaded using the `datasets` library.
2. **Cleaning**: Articles and summaries are cleaned to remove HTML tags, special characters, and extra whitespace.
3. **Splitting**: The dataset is divided into training, validation, and testing subsets.

### Step 2: Extractive Summarization
1. **Sentence Splitting**: Articles are divided into sentences using NLTK.
2. **Sentence Selection**: Sentences are classified using a pretrained BERT model for sequence classification.
3. **Key Sentences**: Selected sentences form the extractive summary.

### Step 3: Abstractive Summarization
1. **Fine-Tuning**: The BART model is fine-tuned on the cleaned dataset using `Seq2SeqTrainer`.
2. **Summary Generation**: Summaries are generated using beam search for improved results.

### Step 4: Evaluation
1. **ROUGE Score**: Measures the overlap of n-grams between generated and reference summaries.
2. **BLEU Score**: Assesses the similarity of generated summaries to reference summaries.
3. **Visualization**: Displays example articles, reference summaries, and generated summaries.

---

## How to Run

1. **Clone the Notebook**
   Save the notebook on your local system or Jupyter environment.

2. **Install Dependencies**
   Ensure all required Python libraries are installed (see requirements above).

3. **Execute the Notebook**
   Run each cell sequentially. Major sections are labeled clearly with comments.

4. **Model Training**
   - If fine-tuning BART, the training may take a few hours based on your hardware.
   - You can reduce the dataset size during training for faster results.

5. **Evaluate the Models**
   View the generated summaries and evaluate their performance using ROUGE and BLEU metrics.

---

## Results

### Example Output
**Article**: 
> "John went to the store to buy groceries. He forgot to bring his wallet but managed to find a way to pay."

**Extractive Summary**: 
> "John went to the store to buy groceries."

**Abstractive Summary**: 
> "John bought groceries without his wallet."

### Metrics
- **ROUGE-1**: Measures overlap of unigrams.
- **ROUGE-2**: Measures overlap of bigrams.
- **ROUGE-L**: Measures the longest common subsequence.
- **BLEU**: Measures n-gram overlap with a smoothing function.

---

## Customization
### Parameters
1. **Dataset Size**: Modify the `train_dataset.select` and `val_dataset.select` calls to use fewer samples for quick testing.
2. **Hyperparameters**: Adjust training arguments like `learning_rate`, `num_train_epochs`, and `batch_size` in `Seq2SeqTrainingArguments`.
3. **Model Selection**: Replace `facebook/bart-base` with other transformer models for experimentation.

---

## Saved Model
The trained model and tokenizer are saved in the `./saved_model` directory. Use these files to reload the model for inference:

---

## Troubleshooting
1. **CUDA Errors**: Ensure PyTorch is installed with GPU support and a compatible CUDA version.
2. **Long Training Time**: Reduce the dataset size or batch size for faster iterations.
3. **Missing Dependencies**: Reinstall missing packages using pip.

---

## References
- [Hugging Face Transformers Documentation](https://huggingface.co/docs/transformers/)
- [CNN/Daily Mail Dataset](https://huggingface.co/datasets/abisee/cnn_dailymail)
- [ROUGE Metric](https://github.com/google-research/google-research/tree/master/rouge)

Enjoy summarizing!
