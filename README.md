# **Tamil-to-Sindhi Machine Translation Model**  
This project develops a **Tamil-to-Sindhi machine translation model** using **pre-trained multilingual models** like **mBART** and **M2M-100**. The model is fine-tuned on the **NTREX_ta_sd_benchmark** dataset to improve translation accuracy.  
 
--- 

## 🚀 **Features**   
✅ Preprocessing of Tamil-Sindhi parallel data  
✅ Fine-tuning with Hugging Face Transformers  
✅ Google Colab GPU acceleration for faster training  
✅ BLEU score evaluation for translation quality   

---

## 🛠 **Tech Stack**  
- Python 🐍  
- Hugging Face Transformers 🤗  
- PyTorch 🔥  
- Google Colab 🚀  

---

## 📂 **Dataset**  
We use the **NTREX_ta_sd_benchmark** dataset, which contains Tamil-Sindhi parallel sentences.  

### **Preprocessing the Dataset**  
The dataset is in **JSON format** and needs to be converted to **TSV** for training.  

```python
import json

# Load the dataset
with open('data.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Convert to tab-separated format
with open('tamil_sindhi.tsv', 'w', encoding='utf-8') as f:
    for entry in data:
        f.write(f"{entry['sourceText']}\t{entry['targetText']}\n")
```

---

## 🔧 **Installation & Setup**  
1️⃣ **Clone the Repository**  
```bash
git clone https://github.com/yourusername/tamil-sindhi-translation.git  
cd tamil-sindhi-translation  
```

2️⃣ **Install Dependencies**  
```bash
pip install transformers datasets evaluate sentencepiece
```

3️⃣ **Prepare the Data**  
```python
from datasets import load_dataset

dataset = load_dataset('csv', data_files='tamil_sindhi.tsv', delimiter='\t', column_names=['source', 'target'])
```

---

## 🎯 **Model Training**  
### **Load Pretrained Model & Tokenizer**  
```python
from transformers import MBart50TokenizerFast, MBartForConditionalGeneration

tokenizer = MBart50TokenizerFast.from_pretrained('facebook/mbart-large-50-many-to-many-mmt')
tokenizer.src_lang = 'ta_IN'  # Tamil  
tokenizer.tgt_lang = 'sd_PK'  # Sindhi  

model = MBartForConditionalGeneration.from_pretrained('facebook/mbart-large-50-many-to-many-mmt')
```

### **Tokenize the Data**  
```python
def preprocess_function(examples):
    inputs = [ex for ex in examples['source']]
    targets = [ex for ex in examples['target']]
    model_inputs = tokenizer(inputs, max_length=128, truncation=True, padding="max_length")

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=128, truncation=True, padding="max_length")

    model_inputs['labels'] = labels['input_ids']
    return model_inputs

tokenized_dataset = dataset.map(preprocess_function, batched=True)
```

### **Set Up Training Configuration**  
```python
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer

training_args = Seq2SeqTrainingArguments(
    output_dir='./results',
    per_device_train_batch_size=4,  
    per_device_eval_batch_size=4,  
    num_train_epochs=1,  
    max_steps=200,  # Limits training for faster completion
    fp16=True,  # Enable mixed precision for faster training
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    tokenizer=tokenizer,
)
```

### **Start Training**  
```python
trainer.train()
```

---

## 📊 **Model Evaluation**  
To evaluate the model, we compute the **BLEU score** to check translation quality.  

### **Install the Evaluation Library**  
```bash
pip install evaluate
```

### **Run BLEU Score Evaluation**  
```python
import evaluate

bleu = evaluate.load("bleu")

# Example Tamil input and expected Sindhi output
translation = "توهان ڪيئن آهيو؟"  # Replace with your model output
reference_translation = ["توهان ڪيئن آهيو؟"]  # Expected Sindhi translation

results = bleu.compute(predictions=[translation], references=[reference_translation])
print(results)
```

---

## 🔥 **Inference: Translate Tamil to Sindhi**  
Once trained, you can use the model for translation.  

```python
model.eval()
sample = "உங்கள் தமிழ் உரை இங்கே"  # Replace with Tamil text
inputs = tokenizer(sample, return_tensors="pt")
translated_tokens = model.generate(**inputs)
translation = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
print(translation)  # Expected Sindhi translation
```

---

## ⚡ **Optimizations for Faster Training**  
If training is slow in Google Colab:  
✅ **Check GPU Type**  
```python
!nvidia-smi
```
✅ **Reduce Training Steps**  
```python
training_args = Seq2SeqTrainingArguments(
    output_dir='./results',
    max_steps=100,  # Reduce training time
    fp16=True,  # Mixed precision for speed
)
```
✅ **Use a Smaller Model**  
Try `Helsinki-NLP/opus-mt-ta-en` instead of `mBART`.  

---

## 📜 **License**  
This project is open-source under the **MIT License**.  

---

## 🤝 **Contributions**  
Feel free to **contribute** or **open issues** if you find bugs!  

---

## 🚀 **Future Work**  
- Fine-tune on **more data** for better accuracy  
- Deploy as an **API for real-time translation**  
- Experiment with **other transformer models**  

---

Let me know if you need **modifications**! 🚀🔥
