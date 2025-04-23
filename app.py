import streamlit as st
import torch
import numpy as np
from transformers import BertTokenizer, BertModel
import torch.nn as nn
import json
import os
import time
import matplotlib.pyplot as plt
import pandas as pd
import sys

# Hack to avoid the torch.classes.__path__._path error
sys.modules['torch.classes'] = None

# Define the model class (must be identical to the one used in training)
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

# Load necessary resources
@st.cache_resource
def load_resources():
    try:
        # Paths
        data_dir = "/Users/migue/Desktop/Maestría/ILN/Proyecto/data"
        emotions_path = os.path.join(data_dir, "emotions.txt")
        ekman_mapping_path = os.path.join(data_dir, "ekman_mapping.json")
        model_path = os.path.join("/Users/migue/Desktop/Maestría/ILN/Proyecto", "./training_002/goemotions_bert_model_lemma.pt")
        
        # Check if files exist
        if not os.path.exists(emotions_path):
            st.error(f"Emotions file does not exist at: {emotions_path}")
            return None
            
        if not os.path.exists(ekman_mapping_path):
            st.error(f"Ekman mapping file does not exist at: {ekman_mapping_path}")
            return None
            
        if not os.path.exists(model_path):
            st.error(f"Model file does not exist at: {model_path}")
            return None
        
        # Load emotions list
        with open(emotions_path, "r") as f:
            emotions = [line.strip() for line in f.readlines()]
        
        # Load Ekman mapping
        with open(ekman_mapping_path, "r") as f:
            ekman_mapping = json.load(f)
        
        # Create mapping from emotion to Ekman category
        emotion_to_ekman = {}
        for ekman_category, emotion_list in ekman_mapping.items():
            for emotion in emotion_list:
                emotion_to_ekman[emotion] = ekman_category
        
        # Add neutral
        emotion_to_ekman["neutral"] = "neutral"
        
        # Load tokenizer
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        
        # Load model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = BertForMultiLabelClassification(len(emotions))
        
        # Try to load the model with different options
        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            
            # Alternative attempt
            try:
                model = torch.load(model_path, map_location=device)
                st.success("Model loaded with alternative method")
            except Exception as e2:
                st.error(f"Error with alternative method: {str(e2)}")
                return None
        
        model.to(device)
        model.eval()
        
        return {
            'emotions': emotions,
            'emotion_to_ekman': emotion_to_ekman,
            'tokenizer': tokenizer,
            'model': model,
            'device': device
        }
    except Exception as e:
        st.error(f"Error loading resources: {str(e)}")
        return None

# Function to predict emotions
def predict_emotions(text, resources, threshold=0.3):
    emotions = resources['emotions']
    emotion_to_ekman = resources['emotion_to_ekman']
    tokenizer = resources['tokenizer']
    model = resources['model']
    device = resources['device']
    
    # Tokenize
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
    start_time = time.time()
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs['logits']
    
    # Convert to probabilities
    probs = logits.cpu().numpy()[0]
    
    # Get predictions above threshold
    pred_indices = np.where(probs > threshold)[0]
    
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
    
    # Sort by probability
    results = sorted(results, key=lambda x: x['probability'], reverse=True)
    
    # Calculate Ekman distribution
    ekman_distribution = {}
    ekman_categories = set(emotion_to_ekman.values())
    for category in ekman_categories:
        cat_emotions = [r for r in results if r['ekman_category'] == category]
        if cat_emotions:
            ekman_distribution[category] = sum(e['probability'] for e in cat_emotions)
    
    return {
        "text": text,
        "emotions": results,
        "top_emotion": results[0] if results else None,
        "ekman_distribution": ekman_distribution,
        "prediction_time": time.time() - start_time
    }

# Create bar chart for emotions
def plot_emotions(prediction):
    emotions = prediction["emotions"][:10]  # Top 10 emotions
    
    if not emotions:
        return None
    
    # Create DataFrame for visualization
    df = pd.DataFrame(emotions)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(df["emotion"], df["probability"], color='skyblue')
    
    # Add Ekman category labels
    for i, bar in enumerate(bars):
        ekman = df.iloc[i]["ekman_category"]
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                f"({ekman})", va='center', fontsize=10, color='gray')
    
    ax.set_xlabel('Probability')
    ax.set_title('Detected Emotions')
    ax.set_xlim(0, 1)
    
    return fig

# Create pie chart for Ekman categories
def plot_ekman(prediction):
    ekman_dist = prediction["ekman_distribution"]
    
    if not ekman_dist:
        return None
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    labels = list(ekman_dist.keys())
    sizes = list(ekman_dist.values())
    
    # Colors for each category
    color_map = {
        'anger': 'red',
        'disgust': 'green',
        'fear': 'purple',
        'joy': 'yellow',
        'sadness': 'blue',
        'surprise': 'orange',
        'neutral': 'gray'
    }
    
    colors = [color_map.get(label, 'lightgray') for label in labels]
    
    ax.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
    ax.axis('equal')
    ax.set_title('Ekman Emotional Categories Distribution')
    
    return fig

# Main application
def main():
    st.set_page_config(
        page_title="Emotion Detector",
        page_icon="😀",
        layout="wide"
    )
    
    st.title("BERT-based Emotion Detector")
    st.write("This application uses a trained BERT model to detect emotions in text.")
    
    # Load resources
    with st.spinner('Loading model and resources...'):
        resources = load_resources()
    
    if resources is None:
        st.error("Failed to load resources. Please check the error messages above.")
        return
    
    # Settings
    with st.sidebar:
        st.header("Settings")
        threshold = st.slider("Detection threshold", 0.0, 1.0, 0.3, 0.05)
    
    # Text input
    text_input = st.text_area("Enter text to analyze", 
                             "I've never been this sad in my life!", 
                             height=150)
    
    # Button to analyze
    if st.button("Analyze Emotions"):
        if text_input.strip():
            with st.spinner('Analyzing emotions...'):
                prediction = predict_emotions(text_input, resources, threshold)
            
            # Show results
            st.subheader("Results")
            
            # General information
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**Prediction time:** {prediction['prediction_time']:.3f} seconds")
                
                if prediction["top_emotion"]:
                    st.write(f"**Top emotion:** {prediction['top_emotion']['emotion']} ({prediction['top_emotion']['probability']:.3f})")
                else:
                    st.write("**No emotions detected** above the threshold.")
            
            # Show detailed results in two columns
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Detected Emotions")
                emotion_fig = plot_emotions(prediction)
                if emotion_fig:
                    st.pyplot(emotion_fig)
                else:
                    st.write("No emotions detected above the threshold.")
            
            with col2:
                st.subheader("Ekman Categories Distribution")
                ekman_fig = plot_ekman(prediction)
                if ekman_fig:
                    st.pyplot(ekman_fig)
                else:
                    st.write("No Ekman emotional categories detected.")
            
            # Detailed emotions table
            st.subheader("Emotion Details")
            if prediction["emotions"]:
                emotions_df = pd.DataFrame(prediction["emotions"])
                st.dataframe(emotions_df)
            else:
                st.write("No emotions detected above the threshold.")
        else:
            st.error("Please enter text to analyze.")

if __name__ == "__main__":
    main()