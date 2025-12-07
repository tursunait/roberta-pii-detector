import torch
import json
import numpy as np
from transformers import AutoModelForTokenClassification, AutoTokenizer
from seqeval.metrics import precision_score, recall_score, f1_score, classification_report

print("="*70)
print("🚀 REDDIT EVALUATION - Using Manually Labeled Data")
print("="*70)

# ============================================================================
# STEP 1: Load her manually annotated Reddit data
# ============================================================================

print("\n📥 Step 1: Loading manually annotated Reddit data...")

REDDIT_DATA_FILE = "./evaluation/reddit_manual_annotations_COMPLETED.json"

try:
    with open(REDDIT_DATA_FILE, 'r') as f:
        reddit_data = json.load(f)
    print(f"✅ Loaded {len(reddit_data)} annotated Reddit posts")
    print(f"   File: {REDDIT_DATA_FILE}")
except Exception as e:
    print(f"❌ Error loading data: {e}")
    print(f"   Make sure file exists at: {REDDIT_DATA_FILE}")
    exit(1)

# Show sample
print(f"\n📄 Sample post structure:")
sample = reddit_data[0]
print(f"   Keys: {list(sample.keys())}")
print(f"   Text length: {len(sample.get('text', ''))} characters")
print(f"   True entities: {len(sample.get('true_entities', []))}")

# ============================================================================
# STEP 2: Load YOUR trained model
# ============================================================================

print("\n🤖 Step 2: Loading YOUR trained model...")

MODEL_PATH = "./model/my_trained_pii_model"

try:
    model = AutoModelForTokenClassification.from_pretrained(MODEL_PATH)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"✅ Model loaded successfully")
    print(f"   Path: {MODEL_PATH}")
    print(f"   Labels: {len(model.config.id2label)}")
    print(f"   Device: {device}")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    exit(1)

# ============================================================================
# STEP 3: Helper function to convert entities to BILOU labels
# ============================================================================

def entities_to_bilou_labels(text, entities, tokenizer):
    """Convert character-level entities to token-level BILOU labels"""
    
    # Tokenize
    encoding = tokenizer(
        text, 
        return_offsets_mapping=True, 
        truncation=True, 
        max_length=512,
        padding=False
    )
    
    # Initialize labels
    labels = ['O'] * len(encoding['input_ids'])
    offset_mapping = encoding['offset_mapping']
    
    # Map each entity to tokens
    for entity in entities:
        entity_start = entity['start']
        entity_end = entity['end']
        entity_label = entity['label']
        
        # Find overlapping tokens
        token_indices = []
        for idx, (token_start, token_end) in enumerate(offset_mapping):
            # Skip special tokens
            if token_start == token_end == 0:
                continue
            # Check overlap
            if token_start < entity_end and token_end > entity_start:
                token_indices.append(idx)
        
        # Apply BILOU tags
        if len(token_indices) == 1:
            labels[token_indices[0]] = f'U-{entity_label}'
        elif len(token_indices) > 1:
            labels[token_indices[0]] = f'B-{entity_label}'
            for idx in token_indices[1:-1]:
                labels[idx] = f'I-{entity_label}'
            labels[token_indices[-1]] = f'L-{entity_label}'
    
    return labels, encoding

# ============================================================================
# STEP 4: Run predictions and compare to ground truth
# ============================================================================

print("\n🔮 Step 3: Running predictions and comparing to ground truth...")

all_true_labels = []
all_pred_labels = []
detailed_results = []

for i, sample in enumerate(reddit_data):
    text = sample['text']
    true_entities = sample.get('true_entities', [])
    source = sample.get('source', 'unknown')
    
    # Get ground truth BILOU labels
    true_labels, encoding = entities_to_bilou_labels(text, true_entities, tokenizer)
    
    # Get model predictions
    inputs = {k: torch.tensor([v]).to(device) for k, v in encoding.items() if k != 'offset_mapping'}
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    predictions = torch.argmax(outputs.logits, dim=-1)[0].cpu().tolist()
    pred_labels = [model.config.id2label.get(p, 'O') for p in predictions]
    
    # Extract predicted entities for display
    predicted_entities = []
    current_entity = None
    offset_mapping = encoding['offset_mapping']
    
    for idx, (pred_label, (start, end)) in enumerate(zip(pred_labels, offset_mapping)):
        if start == end == 0:
            continue
        
        if pred_label == 'O':
            if current_entity:
                predicted_entities.append(current_entity)
                current_entity = None
        elif pred_label.startswith('U-'):
            if current_entity:
                predicted_entities.append(current_entity)
            predicted_entities.append({
                'start': start,
                'end': end,
                'label': pred_label[2:],
                'text': text[start:end]
            })
            current_entity = None
        elif pred_label.startswith('B-'):
            if current_entity:
                predicted_entities.append(current_entity)
            current_entity = {
                'start': start,
                'end': end,
                'label': pred_label[2:],
                'text': text[start:end]
            }
        elif pred_label.startswith(('I-', 'L-')):
            if current_entity:
                current_entity['end'] = end
                current_entity['text'] = text[current_entity['start']:end]
                if pred_label.startswith('L-'):
                    predicted_entities.append(current_entity)
                    current_entity = None
    
    if current_entity:
        predicted_entities.append(current_entity)
    
    # Filter out special tokens
    filtered_true = []
    filtered_pred = []
    
    for j, (start, end) in enumerate(offset_mapping):
        if start == end == 0:
            continue
        filtered_true.append(true_labels[j])
        filtered_pred.append(pred_labels[j])
    
    all_true_labels.append(filtered_true)
    all_pred_labels.append(filtered_pred)
    
    # Store results
    detailed_results.append({
        'id': i,
        'text': text,
        'source': source,
        'true_entities': true_entities,
        'predicted_entities': predicted_entities,
        'true_count': len(true_entities),
        'pred_count': len(predicted_entities)
    })
    
    # Show first few examples
    if i < 3:
        print(f"\n📄 Post {i+1}:")
        print(f"   Text: {text[:100]}...")
        print(f"   True PII: {len(true_entities)} entities")
        for ent in true_entities:
            print(f"      ✓ {ent['label']}: '{ent.get('text', text[ent['start']:ent['end']])}'")
        print(f"   Predicted: {len(predicted_entities)} entities")
        for ent in predicted_entities:
            print(f"      → {ent['label']}: '{ent['text']}'")

# ============================================================================
# STEP 5: Calculate metrics
# ============================================================================

print("\n" + "="*70)
print("📊 EVALUATION METRICS")
print("="*70)

try:
    precision = precision_score(all_true_labels, all_pred_labels)
    recall = recall_score(all_true_labels, all_pred_labels)
    f1 = f1_score(all_true_labels, all_pred_labels)
    
    print(f"\n🎯 Overall Performance:")
    print(f"   Precision: {precision:.4f} ({precision*100:.2f}%)")
    print(f"   Recall:    {recall:.4f} ({recall*100:.2f}%)")
    print(f"   F1 Score:  {f1:.4f} ({f1*100:.2f}%)")
    
    print(f"\n📋 Detailed Classification Report:")
    print(classification_report(all_true_labels, all_pred_labels))
    
except Exception as e:
    print(f"⚠️  Error calculating metrics: {e}")
    print("This might happen if there are no entities or label mismatches")

# ============================================================================
# STEP 6: Analyze errors
# ============================================================================

print("\n" + "="*70)
print("🔍 ERROR ANALYSIS")
print("="*70)

# Calculate per-sample accuracy
correct_samples = 0
total_true_entities = sum(r['true_count'] for r in detailed_results)
total_pred_entities = sum(r['pred_count'] for r in detailed_results)

print(f"\n📈 Entity Counts:")
print(f"   Total true entities: {total_true_entities}")
print(f"   Total predicted entities: {total_pred_entities}")
print(f"   Difference: {total_pred_entities - total_true_entities:+d}")

# Find samples with biggest discrepancies
print(f"\n⚠️  Samples with Biggest Discrepancies:")
discrepancies = sorted(detailed_results, 
                      key=lambda x: abs(x['pred_count'] - x['true_count']), 
                      reverse=True)[:3]

for i, sample in enumerate(discrepancies, 1):
    diff = sample['pred_count'] - sample['true_count']
    print(f"\n{i}. Post {sample['id']+1}: {diff:+d} entities")
    print(f"   True: {sample['true_count']}, Predicted: {sample['pred_count']}")
    print(f"   Text: {sample['text'][:100]}...")

# Count by entity type
true_type_counts = {}
pred_type_counts = {}

for result in detailed_results:
    for ent in result['true_entities']:
        label = ent['label']
        true_type_counts[label] = true_type_counts.get(label, 0) + 1
    for ent in result['predicted_entities']:
        label = ent['label']
        pred_type_counts[label] = pred_type_counts.get(label, 0) + 1

print(f"\n🏷️  Entity Type Breakdown:")
all_types = set(list(true_type_counts.keys()) + list(pred_type_counts.keys()))
print(f"{'Type':<15} {'True':>6} {'Predicted':>10} {'Diff':>6}")
print("-" * 40)
for entity_type in sorted(all_types):
    true_count = true_type_counts.get(entity_type, 0)
    pred_count = pred_type_counts.get(entity_type, 0)
    diff = pred_count - true_count
    print(f"{entity_type:<15} {true_count:>6} {pred_count:>10} {diff:>+6}")

# ============================================================================
# STEP 7: Save results
# ============================================================================

print("\n" + "="*70)
print("💾 SAVING RESULTS")
print("="*70)

output = {
    'model_path': MODEL_PATH,
    'test_data': REDDIT_DATA_FILE,
    'num_samples': len(reddit_data),
    'metrics': {
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1)
    },
    'entity_counts': {
        'true_total': total_true_entities,
        'pred_total': total_pred_entities,
        'by_type_true': true_type_counts,
        'by_type_pred': pred_type_counts
    },
    'detailed_results': detailed_results
}

output_file = 'my_reddit_evaluation_results.json'
with open(output_file, 'w') as f:
    json.dump(output, f, indent=2)

print(f"✅ Results saved to: {output_file}")

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("\n" + "="*70)
print("✅ EVALUATION COMPLETE!")
print("="*70)

print(f"\n🎯 Final Summary:")
print(f"   • Evaluated on {len(reddit_data)} Reddit posts")
print(f"   • F1 Score: {f1:.4f} ({f1*100:.1f}%)")
print(f"   • Precision: {precision:.4f}")
print(f"   • Recall: {recall:.4f}")

if f1 > 0.7:
    print(f"\n🎉 Great performance! F1 > 70%")
elif f1 > 0.5:
    print(f"\n👍 Decent performance, room for improvement")
else:
    print(f"\n⚠️  Performance needs improvement")

print("\n" + "="*70)