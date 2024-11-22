# PRODIGY_GA_01
TEXT GENERATION WITH GPT-2
---

# **Fine-Tuning GPT-2 for Contextual Text Generation**

## **Overview**  
This project is part of my internship task, **Generative AI Code Prodigy_GA_01** with Prodigy InfoTech. The objective was to fine-tune **GPT-2**, a transformer model OpenAI developed to generate coherent and contextually relevant text based on a given prompt. By fine-tuning the model on a custom dataset, the output mimics the style and structure of the training data, showcasing the model's adaptability.

---

## **Project Objectives**  
1. Train GPT-2 to generate text relevant to a given context or style.  
2. Use a custom dataset to teach GPT-2 specific language patterns.  
3. Demonstrate the practical application of fine-tuning transformer models for domain-specific tasks.

---

## **Features**  
- **Custom Dataset**: Model trained on a dataset tailored to specific styles or content.  
- **Fine-Tuned GPT-2**: Adapted the pre-trained GPT-2 for customized text generation.  
- **Text Generation**: Generate text based on user-defined prompts, with results that emulate the training data’s style.

---

## **Setup Instructions**

### 1. **Environment Setup**  
This project was implemented in **Google Colab** for GPU support. Install the required libraries:  
```bash
pip install transformers datasets torch
```

### 2. **Clone the Repository**  
```bash
git clone https://github.com/Nsi20/PRODIGY_GA_01.git
cd PRODIGY_GA_01
```

### 3. **Prepare the Dataset**  
Ensure your custom dataset is in the same directory as the script or specify its path in the code.  
- Format: Plain text file (`.txt`) with each entry on a new line.  
- Example:  
  ```
  Once upon a time, in a distant galaxy...  
  In a world filled with mysteries and adventures...  
  ```

---

## **Fine-Tuning Process**

### **Step 1: Import GPT-2 and Tokenizer**  
```python
from transformers import GPT2Tokenizer, GPT2LMHeadModel

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")
```

### **Step 2: Load and Tokenize the Dataset**  
```python
from datasets import load_dataset

dataset = load_dataset("text", data_files={"train": "custom_data.txt"})
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)

tokenized_datasets = dataset.map(tokenize_function, batched=True)
```

### **Step 3: Fine-Tune the Model**  
Set up the Hugging Face **Trainer** to fine-tune the model:  
```python
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="./fine_tuned_gpt2",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    save_steps=10_000,
    save_total_limit=2
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"]
)
trainer.train()
```

### **Step 4: Generate Text**  
Test the fine-tuned model by generating text for a given prompt:  
```python
prompt = "Once upon a time"
input_ids = tokenizer(prompt, return_tensors="pt").input_ids
output = model.generate(input_ids, max_length=100)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

---

## **Results**  
After training, the model was able to generate text that aligned with the style and structure of the training dataset. Example output:  
```text
Once upon a time in a distant galaxy, the Sun and its stars...
```

---

## **Project Directory Structure**  
```
PRODIGY_GA_01/
│
├── fine_tuned_gpt2/             # Fine-tuned model directory
│   ├── generation_config.json
│   ├── tokenizer_config.json
│   ├── vocab.json
│   ├── merges.txt
│   ├── model.safetensors
│   ├── config.json
│
├── custom_data.txt              # Training dataset
├── fine_tune_gpt2.ipynb         # Jupyter Notebook for the project
└── README.md                    # Project documentation
```

---

## **Key Learnings**  
- **Fine-Tuning GPT-2**: Fine-tuning pre-trained models allows for domain-specific text generation.  
- **Dataset Preparation**: Proper dataset formatting is critical for fine-tuning success.  
- **Model Evaluation**: Analyzing generated text helps measure coherence and contextual accuracy.  

---

## **Future Enhancements**  
1. Fine-tune GPT-2 on larger datasets for improved coherence.  
2. Experiment with other hyperparameters (e.g., learning rate, batch size).  
3. Deploy the fine-tuned model as an API for real-time text generation.

---

## **Acknowledgment**  
Special thanks to **Prodigy Info Tech** for this enriching learning experience as part of the internship program.

---

## **Connect**  
🔗 **GitHub Repository**: [PRODIGY_GA_01](https://github.com/Nsi20/PRODIGY_GA_01)  
💼 **LinkedIn**: [Nsidibe Daniel Essang](https://www.linkedin.com/in/nsidibe-essang-142778204/)  

---

### **Tags**:  
`GPT-2` `Generative AI` `OpenAI` `Hugging Face` `Fine-Tuning` `Machine Learning` `Python` `AI Models` `Internship Projects`
