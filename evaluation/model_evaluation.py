"""
Complete PII Evaluation Script
Evaluates model on real-world PII dataset (ai4privacy/pii-masking-300k)
"""

import torch
import json
from transformers import AutoModelForTokenClassification, AutoTokenizer
from seqeval.metrics import precision_score, recall_score, f1_score, classification_report, accuracy_score
from datasets import load_dataset

print("="*70)
print("🔍 PII MODEL EVALUATION - Real-World Dataset")
print("="*70)

# ============================================================================
# STEP 1: Load real-world PII dataset
# ============================================================================

print("\n📥 Step 1: Loading ai4privacy PII dataset...")

try:
    # Load pre-labeled PII dataset
    dataset = load_dataset("ai4privacy/pii-masking-300k", split="train")

    # ✅ FILTER TO ENGLISH ONLY
    dataset_en = dataset.filter(lambda x: x['language'] == 'English')

    
    # Take first 100 examples for evaluation (adjust as needed)
    test_data = dataset_en.select(range(min(300, len(dataset_en))))
    
    print(f"✅ Loaded {len(test_data)} examples from ai4privacy dataset")
    print(f"   Dataset has real-world text with labeled PII")
    
    # Show sample
    sample = test_data[0]
    print(f"\n📄 Sample structure:")
    print(f"   Keys: {list(sample.keys())}")
    
except Exception as e:
    print(f"❌ Error loading dataset: {e}")
    print(f"\n   Make sure you have internet connection!")
    exit(1)

# ============================================================================
# STEP 2: Load trained model
# ============================================================================

print("\n🤖 Step 2: Loading trained model...")

MODEL_PATH = "./trained_model"

try:
    model = AutoModelForTokenClassification.from_pretrained(MODEL_PATH)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    print(f"✅ Model loaded")
    print(f"   Labels: {len(model.config.id2label)}")
    print(f"   Device: {device}")
    
except Exception as e:
    print(f"❌ Error loading model: {e}")
    print(f"   Make sure model exists at: {MODEL_PATH}")
    exit(1)

# ============================================================================
# STEP 3: Convert dataset format to match our model
# ============================================================================

import json

def convert_dataset_to_our_format(example):
    """
    Convert ai4privacy format to our evaluation format
    """
    text = example['source_text']
    entities = []
    
    if 'span_labels' in example and example['span_labels']:
        try:
            span_labels = eval(example['span_labels'])
            
            for span in span_labels:
                if isinstance(span, (list, tuple)) and len(span) >= 3:
                    start = span[0]
                    end = span[1]
                    label = span[2]
                    
                    # EXPANDED label mapping for train split
                    label_mapping = {
                        # Person names
                        'USERNAME': 'PERSON',
                        'LASTNAME1': 'PERSON',
                        'LASTNAME2': 'PERSON',
                        'LASTNAME3': 'PERSON',
                        'GIVENNAME1': 'PERSON',
                        'GIVENNAME2': 'PERSON',
                        'NAME': 'PERSON',
                        
                        # Contact info
                        'EMAIL': 'EMAIL',
                        'TEL': 'PHONE',
                        
                        # SSN-like
                        'SOCIALNUMBER': 'SSN',
                        
                        # Address components
                        'STREET': 'ADDRESS',
                        'CITY': 'ADDRESS',
                        'STATE': 'ADDRESS',
                        'POSTCODE': 'ADDRESS',
                        'BUILDING': 'ADDRESS',
                        'SECADDRESS': 'ADDRESS',
                        'COUNTRY': 'ADDRESS',
                        
                        # Dates/Times
                        # 'TIME': 'DATE',
                        'DATE': 'DATE',
                        'BOD': 'DATE',  # Birth of Date
                        
                        # Organizations (approximate)
                        'COMPANY': 'ORG',

                    }
                    
                    mapped_label = label_mapping.get(label, None)
                    
                    if mapped_label in ['PERSON', 
                                        'EMAIL', 
                                        'PHONE', 
                                        'ADDRESS', 
                                        'DATE', 
                                        'AGE', 
                                        'ORG', 
                                        'SSN', 
                                        'CREDIT_CARD']:
                        entity_text = text[start:end] if 0 <= start < len(text) and 0 < end <= len(text) else ""
                        
                        entities.append({
                            'start': start,
                            'end': end,
                            'label': mapped_label,
                            'text': entity_text
                        })
        
        except Exception as e:
            pass
    
    return {
        'text': text,
        'true_entities': entities
    }


# ============================================================================
# STEP 4: Helper function - entities to BILOU labels
# ============================================================================

def entities_to_bilou_labels(text, entities, tokenizer):
    """Convert character-level entities to token-level BILOU labels"""
    
    encoding = tokenizer(
        text, 
        return_offsets_mapping=True,
        truncation=True,
        max_length=512,
        padding=False
    )
    
    labels = ['O'] * len(encoding['input_ids'])
    offset_mapping = encoding['offset_mapping']
    
    for entity in entities:
        entity_start = entity['start']
        entity_end = entity['end']
        entity_label = entity['label'].upper()
        
        # Map entity label to our model's labels
        # The dataset might use different label names
        label_mapping = {
            'NAME': 'PERSON',
            'PERSON': 'PERSON',
            'EMAIL': 'EMAIL',
            'PHONE': 'PHONE',
            'PHONE_NUMBER': 'PHONE',
            'ADDRESS': 'ADDRESS',
            'DATE': 'DATE',
            'AGE': 'AGE',
            'SSN': 'SSN',
            'CREDIT_CARD': 'CREDIT_CARD',
            'ORGANIZATION': 'ORG',
            'ORG': 'ORG',
        }
        
        entity_label = label_mapping.get(entity_label, entity_label)
        
        # Skip if label not in our model
        if entity_label not in model.config.label2id and \
           f'B-{entity_label}' not in model.config.label2id:
            continue
        
        token_indices = []
        for idx, (token_start, token_end) in enumerate(offset_mapping):
            if token_start == token_end == 0:
                continue
            if token_start < entity_end and token_end > entity_start:
                token_indices.append(idx)
        
        if len(token_indices) == 1:
            labels[token_indices[0]] = f'U-{entity_label}'
        elif len(token_indices) > 1:
            labels[token_indices[0]] = f'B-{entity_label}'
            for idx in token_indices[1:-1]:
                labels[idx] = f'I-{entity_label}'
            labels[token_indices[-1]] = f'L-{entity_label}'
    
    return labels, encoding

# ============================================================================
# STEP 5: Run predictions on dataset
# ============================================================================

print("\n🔮 Step 3: Running predictions on PII dataset...")

all_true_labels = []
all_pred_labels = []
results = []

for i, example in enumerate(test_data):
    # Convert format
    formatted = convert_dataset_to_our_format(example)
    text = formatted['text']
    true_entities = formatted['true_entities']
    
    # Skip if text is too short or empty
    if not text or len(text) < 10:
        continue
    
    try:
        # Get ground truth labels
        true_labels, encoding = entities_to_bilou_labels(text, true_entities, tokenizer)
        
        # Get predictions
        inputs = {k: torch.tensor([v]).to(device) for k, v in encoding.items() if k != 'offset_mapping'}
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        # ===== CONFIDENCE THRESHOLD FILTERING =====
        logits = outputs.logits  # Shape: [1, seq_len, num_labels]
        probs = torch.softmax(logits, dim=-1)  # Convert to probabilities
        max_probs, raw_predictions = torch.max(probs, dim=-1)  # Get max prob and prediction
        
        # Apply confidence threshold
        confidence_threshold = 0.3  # Tune this: 0.5 (lenient) to 0.8 (strict)
        predictions_list = raw_predictions[0].cpu().tolist()
        confidences_list = max_probs[0].cpu().tolist()
        
        # Filter low-confidence predictions
        filtered_predictions = []
        for pred, conf in zip(predictions_list, confidences_list):
            if conf < confidence_threshold and pred != 0:  # If low confidence AND not 'O'
                filtered_predictions.append(0)  # Change to 'O'
            else:
                filtered_predictions.append(pred)  # Keep prediction
        
        # Convert predictions to labels
        pred_labels = [model.config.id2label.get(p, 'O') for p in filtered_predictions]
        
        # Filter out special tokens
        filtered_true = []
        filtered_pred = []
        for j, (start, end) in enumerate(encoding['offset_mapping']):
            if start == end == 0:
                continue
            filtered_true.append(true_labels[j])
            filtered_pred.append(pred_labels[j])
        
        all_true_labels.append(filtered_true)
        all_pred_labels.append(filtered_pred)
        
        results.append({
            'id': i,
            'text': text[:200],  # Store first 200 chars
            'true_count': len(true_entities),
            'pred_count': sum(1 for label in filtered_pred if label != 'O')
        })
        
        # Show progress
        if (i + 1) % 20 == 0:
            print(f"   Processed {i + 1}/{len(test_data)} examples...")
    
    except Exception as e:
        print(f"   ⚠️  Skipped example {i}: {e}")
        continue

print(f"\n✅ Processed {len(results)} examples successfully")
        
    

# ============================================================================
# STEP 6: Calculate metrics
# ============================================================================

print("\n" + "="*70)
print("📊 EVALUATION RESULTS")
print("="*70)

if len(all_true_labels) == 0:
    print("❌ No valid examples to evaluate!")
    exit(1)

try:
    
    precision = precision_score(all_true_labels, all_pred_labels)
    recall = recall_score(all_true_labels, all_pred_labels)
    f1 = f1_score(all_true_labels, all_pred_labels)
    accuracy = accuracy_score(all_true_labels, all_pred_labels) 
    
    print(f"\n🎯 Overall Performance:")
    print(f"   Accuracy:  {accuracy:.4f} ({accuracy*100:.1f}%)") 
    print(f"   Precision: {precision:.4f} ({precision*100:.1f}%)")
    print(f"   Recall:    {recall:.4f} ({recall*100:.1f}%)")
    print(f"   F1 Score:  {f1:.4f} ({f1*100:.1f}%)")

    
    print(f"\n📋 Detailed Classification Report:")
    print(classification_report(all_true_labels, all_pred_labels))
    
except Exception as e:
    print(f"⚠️  Error calculating metrics: {e}")
    f1 = 0.0

# ============================================================================
# STEP 7: Error analysis
# ============================================================================

print("\n" + "="*70)
print("🔍 ERROR ANALYSIS")
print("="*70)

total_true = sum(r['true_count'] for r in results)
total_pred = sum(r['pred_count'] for r in results)

print(f"\n📊 Entity Counts:")
print(f"   True entities: {total_true}")
print(f"   Predicted: {total_pred}")
print(f"   Difference: {total_pred - total_true:+d}")

# ============================================================================
# STEP 8: Save results
# ============================================================================

print("\n💾 Saving results...")

output = {
    'model': MODEL_PATH,
    'test_dataset': 'ai4privacy/pii-masking-300k',
    'num_examples': len(results),
    'metrics': {
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'accuracy': float(accuracy),
    },
    'entity_counts': {
        'true_total': total_true,
        'predicted_total': total_pred
    }
}


with open('evaluation/evaluation_results.json', 'w') as f:
    json.dump(output, f, indent=2)

print("✅ Results saved: evaluation_results.json")
