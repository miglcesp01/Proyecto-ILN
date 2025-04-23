import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
from torch.optim import AdamW  # Importando AdamW desde torch.optim en lugar de transformers
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import classification_report
import json
import os
import logging
import time
from datetime import datetime
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Download NLTK resources (run once)
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt')
    nltk.download('wordnet')

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Function to preprocess text with lemmatization
def preprocess_text(text):
    # Tokenize
    tokens = word_tokenize(text.lower())
    # Lemmatize
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    # Join back to text
    processed_text = ' '.join(lemmatized_tokens)
    return processed_text

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using {device} device")

# Paths
DATA_DIR = "/Users/migue/Desktop/Maestría/ILN/Proyecto/data"
EMOTIONS_PATH = os.path.join(DATA_DIR, "emotions.txt")
EKMAN_MAPPING_PATH = os.path.join(DATA_DIR, "ekman_mapping.json")
TRAIN_PATH = os.path.join(DATA_DIR, "train.tsv")
DEV_PATH = os.path.join(DATA_DIR, "dev.tsv")
TEST_PATH = os.path.join(DATA_DIR, "test.tsv")

# Load emotions list
logger.info("Loading emotions list...")
with open(EMOTIONS_PATH, "r") as f:
    emotions = [line.strip() for line in f.readlines()]
logger.info(f"Loaded {len(emotions)} emotions")

# Create a dictionary mapping emotion indices to emotion names
emotion_idx_to_name = {i: emotion for i, emotion in enumerate(emotions)}

# Load Ekman mapping
logger.info("Loading Ekman mapping...")
with open(EKMAN_MAPPING_PATH, "r") as f:
    ekman_mapping = json.load(f)

# Create a reverse mapping from emotion to ekman category
emotion_to_ekman = {}
for ekman_category, emotion_list in ekman_mapping.items():
    for emotion in emotion_list:
        emotion_to_ekman[emotion] = ekman_category

# Add neutral to the mapping
emotion_to_ekman["neutral"] = "neutral"
logger.info(f"Ekman mapping created with {len(ekman_mapping)} categories")

# Create a mapping from emotion index to ekman category
emotion_idx_to_ekman = {i: emotion_to_ekman.get(emotion, "unknown") 
                        for i, emotion in enumerate(emotions)}

# Define dataset class
class GoEmotionsDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=128, apply_lemmatization=True):
        # Load data
        logger.info(f"Loading dataset from {data_path}...")
        start_time = time.time()
        self.data = pd.read_csv(data_path, sep='\t', header=None)
        self.data.columns = ['text', 'emotions', 'id']
        
        # Process text
        if apply_lemmatization:
            logger.info("Applying lemmatization to texts...")
            lemma_start_time = time.time()
            self.texts = [preprocess_text(text) for text in self.data['text'].tolist()]
            logger.info(f"Lemmatization completed in {time.time() - lemma_start_time:.2f} seconds")
        else:
            self.texts = self.data['text'].tolist()
        
        # Process labels
        self.emotion_lists = []
        for emotion_str in self.data['emotions']:
            if pd.isna(emotion_str):
                self.emotion_lists.append([])
            else:
                self.emotion_lists.append([int(e) for e in emotion_str.split(',')])
        
        # Prepare for multi-label classification
        logger.info("Preparing multi-label binarizer...")
        self.mlb = MultiLabelBinarizer(classes=list(range(len(emotions))))
        self.mlb.fit([list(range(len(emotions)))])
        self.labels = self.mlb.transform(self.emotion_lists)
        
        # BERT tokenizer
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        logger.info(f"Dataset loaded with {len(self.texts)} examples in {time.time() - start_time:.2f} seconds")
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        labels = self.labels[idx]
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Remove batch dimension added by the tokenizer
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        item['labels'] = torch.tensor(labels, dtype=torch.float)
        
        return item

# Define the BERT model for multi-label classification
class BertForMultiLabelClassification(nn.Module):
    def __init__(self, num_labels):
        super(BertForMultiLabelClassification, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, input_ids, attention_mask, token_type_ids=None, labels=None):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        output = self.sigmoid(logits)
        
        loss = None
        if labels is not None:
            loss_fct = nn.BCELoss()
            loss = loss_fct(output, labels)
            
        return {"loss": loss, "logits": output} if loss is not None else {"logits": output}

# Prepare tokenizer
logger.info("Loading BERT tokenizer...")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
logger.info("Tokenizer loaded successfully")

# Function to prepare data
def prepare_dataloader(data_path, batch_size=16, apply_lemmatization=True):
    dataset = GoEmotionsDataset(data_path, tokenizer, apply_lemmatization=apply_lemmatization)
    shuffle = (data_path == TRAIN_PATH)
    logger.info(f"Creating DataLoader with batch_size={batch_size}, shuffle={shuffle}, lemmatization={apply_lemmatization}")
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

# Development mode - use dev.tsv for training to speed up the process
DEV_MODE = False  # Set this to False for full training

# Training function
def train_model(model, train_dataloader, val_dataloader, epochs=3, learning_rate=2e-5):
    logger.info(f"Starting training with learning_rate={learning_rate}, epochs={epochs}")
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    
    for epoch in range(epochs):
        epoch_start_time = time.time()
        logger.info(f"========== Epoch {epoch + 1}/{epochs} ==========")
        
        # Training
        model.train()
        train_loss = 0
        batch_count = len(train_dataloader)
        logger.info(f"Training on {batch_count} batches")
        
        for batch_idx, batch in enumerate(train_dataloader):
            if batch_idx % 10 == 0:
                logger.info(f"Training batch {batch_idx+1}/{batch_count}")
                
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs['loss']
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            if batch_idx % 20 == 0:
                logger.info(f"Batch {batch_idx+1}/{batch_count}, Loss: {loss.item():.4f}")
        
        avg_train_loss = train_loss / len(train_dataloader)
        logger.info(f"Average training loss: {avg_train_loss:.4f}")
        
        # Validation
        logger.info("Starting validation...")
        model.eval()
        val_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_dataloader):
                if batch_idx % 10 == 0:
                    logger.info(f"Validation batch {batch_idx+1}/{len(val_dataloader)}")
                    
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs['loss']
                logits = outputs['logits']
                
                val_loss += loss.item()
                
                # Convert probabilities to binary predictions (threshold 0.5)
                preds = (logits > 0.5).float().cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())
        
        avg_val_loss = val_loss / len(val_dataloader)
        logger.info(f"Average validation loss: {avg_val_loss:.4f}")
        
        # Calculate metrics
        logger.info("Calculating metrics...")
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        # Macro F1 score
        f1_scores = []
        for i in range(len(emotions)):
            try:
                from sklearn.metrics import f1_score
                class_f1 = f1_score(all_labels[:, i], all_preds[:, i])
                f1_scores.append(class_f1)
                logger.info(f"{emotions[i]}: F1 = {class_f1:.4f}")
            except Exception as e:
                logger.warning(f"Could not calculate F1 for {emotions[i]}: {str(e)}")
        
        macro_f1 = np.mean(f1_scores)
        logger.info(f"Macro F1: {macro_f1:.4f}")
        logger.info(f"Epoch completed in {time.time() - epoch_start_time:.2f} seconds")
    
    logger.info("Training completed!")
    return model

# Function to predict emotions for new text
def predict_emotions(model, text, threshold=0.3, apply_lemmatization=True):
    logger.info(f"Predicting emotions for text: '{text[:50]}...' with threshold={threshold}")
    model.eval()
    
    # Apply lemmatization if enabled
    if apply_lemmatization:
        logger.info("Applying lemmatization to input text")
        original_text = text
        text = preprocess_text(text)
        logger.info(f"Original: '{original_text[:50]}...'")
        logger.info(f"Lemmatized: '{text[:50]}...'")
    
    # Tokenize
    logger.info("Tokenizing input text...")
    encoding = tokenizer(
        text,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    # Move to device
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    # Get predictions
    logger.info("Running prediction with model...")
    start_time = time.time()
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs['logits']
    
    # Convert to probabilities
    probs = logits.cpu().numpy()[0]
    
    # Get predictions above threshold
    pred_indices = np.where(probs > threshold)[0]
    logger.info(f"Found {len(pred_indices)} emotions above threshold {threshold}")
    
    # Create result
    results = []
    for idx in pred_indices:
        emotion = emotions[idx]
        ekman_category = emotion_to_ekman.get(emotion, "unknown")
        prob = float(probs[idx])
        results.append({
            "emotion": emotion,
            "ekman_category": ekman_category,
            "probability": prob
        })
        logger.info(f"Detected emotion: {emotion} ({ekman_category}) with probability {prob:.4f}")
    
    # Sort by probability
    results = sorted(results, key=lambda x: x['probability'], reverse=True)
    
    # Format overall results
    formatted_results = {
        "text": text,
        "original_text": original_text if apply_lemmatization else text,
        "lemmatized": apply_lemmatization,
        "emotions": results,
        "top_emotion": results[0] if results else None,
        "ekman_distribution": {}
    }
    
    # Calculate Ekman distribution
    ekman_categories = set(emotion_to_ekman.values())
    for category in ekman_categories:
        cat_emotions = [r for r in results if r['ekman_category'] == category]
        if cat_emotions:
            formatted_results["ekman_distribution"][category] = sum(e['probability'] for e in cat_emotions)
    
    logger.info(f"Prediction completed in {time.time() - start_time:.2f} seconds")
    if results:
        logger.info(f"Top emotion: {formatted_results['top_emotion']['emotion']} with probability {formatted_results['top_emotion']['probability']:.4f}")
    else:
        logger.info("No emotions detected above threshold")
    
    return formatted_results

# Main execution
def main(apply_lemmatization=True):
    logger.info("=== Starting Emotion Classification Training ===")
    logger.info(f"Text preprocessing: Lemmatization is {'enabled' if apply_lemmatization else 'disabled'}")
    start_time = time.time()
    
    # Prepare dataloaders
    logger.info("Preparing dataloaders...")
    if DEV_MODE:
        logger.info("DEVELOPMENT MODE: Using dev.tsv for training to speed up the process")
        train_dataloader = prepare_dataloader(DEV_PATH, batch_size=16, apply_lemmatization=apply_lemmatization)
        val_dataloader = prepare_dataloader(DEV_PATH, batch_size=16, apply_lemmatization=apply_lemmatization)  # Using same data for validation in dev mode
    else:
        logger.info("PRODUCTION MODE: Using full training dataset")
        train_dataloader = prepare_dataloader(TRAIN_PATH, batch_size=16, apply_lemmatization=apply_lemmatization)
        val_dataloader = prepare_dataloader(DEV_PATH, batch_size=16, apply_lemmatization=apply_lemmatization)
    
    # Initialize model
    logger.info("Initializing BERT model for multi-label classification...")
    model = BertForMultiLabelClassification(len(emotions))
    model.to(device)
    logger.info(f"Model initialized with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Train model
    logger.info("Starting model training...")
    epochs = 2 if DEV_MODE else 3  # Use fewer epochs in dev mode
    model = train_model(model, train_dataloader, val_dataloader, epochs=epochs)
    
    # Save model
    logger.info("Saving trained model...")
    lemma_suffix = "_lemma" if apply_lemmatization else ""
    model_path = f"goemotions_bert_model_dev{lemma_suffix}.pt" if DEV_MODE else f"goemotions_bert_model{lemma_suffix}.pt"
    torch.save(model.state_dict(), model_path)
    logger.info(f"Model saved to {model_path}")
    
    # Example of prediction
    test_text = "I've never been this sad in my life!"
    logger.info(f"\nTesting model with example: '{test_text}'")
    prediction = predict_emotions(model, test_text, apply_lemmatization=apply_lemmatization)
    logger.info("Prediction result:")
    logger.info(json.dumps(prediction, indent=2))
    
    # Log total execution time
    total_time = time.time() - start_time
    logger.info(f"=== Emotion Classification Training Completed in {total_time:.2f} seconds ({total_time/60:.2f} minutes) ===")

if __name__ == "__main__":
    # Set to True to enable lemmatization, False to disable
    apply_lemmatization = True
    main(apply_lemmatization)