import torch
import requests
import time
import json
import pandas as pd
from transformers import AutoTokenizer, AutoModelForTokenClassification
from my_trained_pii_model.config_and_labels import LABEL2ID, ID2LABEL

print("🚀 Starting Real-World Evaluation with YOUR Trained Model")

# Load YOUR trained model
model_path = "./my_trained_pii_model"

try:
    print(f"🔧 Loading your model from: {model_path}")
    model = AutoModelForTokenClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    print("✅ Successfully loaded YOUR trained model!")
    
except Exception as e:
    print(f"❌ Error loading model: {e}")
    print("💡 Falling back to trainer object...")
    
    if 'trainer' in globals():
        model = trainer.model
        tokenizer = trainer.tokenizer
        print("✅ Using model from trainer object")
    else:
        raise Exception("No trained model found. Please run training first.")

# STEP 1: Collect Real Reddit Data
def collect_reddit_samples(subreddits=['relationships', 'personalfinance', 'jobs', 'askreddit'], limit=20):
    """
    Collect real text samples from Reddit where people might share personal info
    """
    samples = []
    
    for subreddit in subreddits:
        try:
            print(f"📡 Fetching from r/{subreddit}...")
            url = f"https://www.reddit.com/r/{subreddit}/hot.json?limit={limit}"
            headers = {'User-Agent': 'PII-Detection-Research-Bot/1.0'}
            response = requests.get(url, headers=headers, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                for post in data['data']['children']:
                    post_data = post['data']
                    # Combine title and text
                    text = f"{post_data['title']}. {post_data.get('selftext', '')}"
                    
                    # Filter out very short or removed posts
                    if len(text) > 50 and text not in ['[removed]', '[deleted]']:
                        samples.append({
                            'text': text[:1000],  # Limit length
                            'source': f'reddit_{subreddit}',
                            'post_id': post_data['id'],
                            'true_entities': []  # To be annotated manually
                        })
            else:
                print(f"⚠️  Could not fetch from r/{subreddit}: Status {response.status_code}")
                
            time.sleep(1)  # Be respectful to Reddit's API
            
        except Exception as e:
            print(f"❌ Error fetching from r/{subreddit}: {e}")
    
    print(f"✅ Collected {len(samples)} samples from Reddit")
    return samples

# STEP 2: Create synthetic samples for known ground truth
def create_synthetic_test_samples():
    """Create samples with known PII for reliable metrics"""
    return [
        {
            'text': "Contact me at john.doe@company.com or 555-123-4567 for details",
            'true_entities': [
                {'start': 15, 'end': 34, 'label': 'EMAIL', 'text': 'john.doe@company.com'},
                {'start': 38, 'end': 50, 'label': 'PHONE', 'text': '555-123-4567'}
            ],
            'source': 'synthetic_contact',
            'category': 'contact_info'
        },
        {
            'text': "My social security number is 123-45-6789 and I was born on 1990-05-15",
            'true_entities': [
                {'start': 28, 'end': 39, 'label': 'SSN', 'text': '123-45-6789'},
                {'start': 58, 'end': 68, 'label': 'DATE', 'text': '1990-05-15'}
            ],
            'source': 'synthetic_gov',
            'category': 'government_id'
        }
    ]

# STEP 3: Advanced prediction function
def predict_entities_advanced(texts, model, tokenizer, id2label):
    """Use your trained model for entity prediction with proper token alignment"""
    all_predictions = []
    
    for text in texts:
        try:
            # Tokenize with return_offsets_mapping to get character positions
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, 
                              return_offsets_mapping=True, padding=True)
            
            # Get offset mapping (character positions for each token)
            offset_mapping = inputs.pop('offset_mapping')[0]
            
            # Move to model device
            device = next(model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Predict
            with torch.no_grad():
                outputs = model(**inputs)
            
            # Get predictions
            predictions = torch.argmax(outputs.logits, dim=2)[0].cpu().numpy()
            tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
            
            # Extract entities with proper character positions
            entities = []
            current_entity = None
            
            for i, (token, pred_id, offset) in enumerate(zip(tokens, predictions, offset_mapping)):
                label = id2label.get(pred_id, "O")
                start_char, end_char = offset
                
                # Skip special tokens and padding
                if start_char == end_char == 0:
                    if current_entity:
                        entities.append(current_entity)
                        current_entity = None
                    continue
                
                if label.startswith('B-'):
                    # Start new entity
                    if current_entity:
                        entities.append(current_entity)
                    current_entity = {
                        'text': text[start_char:end_char],
                        'label': label[2:],
                        'start': start_char,
                        'end': end_char
                    }
                elif label.startswith('I-') and current_entity and current_entity['label'] == label[2:]:
                    # Continue entity - extend the text and end position
                    current_entity['text'] += text[start_char:end_char]
                    current_entity['end'] = end_char
                elif current_entity:
                    # End entity
                    entities.append(current_entity)
                    current_entity = None
            
            # Add final entity if exists
            if current_entity:
                entities.append(current_entity)
            
            all_predictions.append({
                'text': text,
                'predicted_entities': entities,
                'tokens': tokens,
                'labels': [id2label.get(pid, "O") for pid in predictions]
            })
            
        except Exception as e:
            print(f"⚠️  Error processing text: {e}")
            all_predictions.append({
                'text': text,
                'predicted_entities': [],
                'tokens': [],
                'labels': []
            })
    
    return all_predictions

# STEP 4: Collect Real Reddit Data
print("\n📡 Collecting real-world data from Reddit...")
reddit_samples = collect_reddit_samples(limit=15)  # Get 15 from each subreddit

# STEP 5: Add synthetic samples for reliable metrics
synthetic_samples = create_synthetic_test_samples()

# STEP 6: Combine all test samples
all_test_samples = synthetic_samples + reddit_samples[:20]  # Use first 20 Reddit samples

print(f"\n📊 Test Dataset Composition:")
print(f"   - Synthetic samples: {len(synthetic_samples)}")
print(f"   - Reddit samples: {len(reddit_samples[:20])}")
print(f"   - Total samples: {len(all_test_samples)}")

# STEP 7: Run predictions on all samples
print("\n🔮 Running model predictions on real-world data...")
texts = [sample['text'] for sample in all_test_samples]
predictions = predict_entities_advanced(texts, model, tokenizer, ID2LABEL)

# STEP 8: Calculate metrics (only for synthetic samples where we have ground truth)
def calculate_metrics_with_reddit(results, synthetic_count):
    """Calculate metrics for synthetic data and analyze Reddit data qualitatively"""
    
    # Metrics for synthetic data (where we have ground truth)
    synthetic_results = results[:synthetic_count]
    reddit_results = results[synthetic_count:]
    
    metrics = {
        'synthetic_metrics': {'tp': 0, 'fp': 0, 'fn': 0},
        'reddit_analysis': {
            'total_samples': len(reddit_results),
            'samples_with_pii': 0,
            'total_entities_detected': 0,
            'entities_by_type': {}
        }
    }
    
    # Calculate synthetic metrics
    for result in synthetic_results:
        true_entities = set((e['text'].lower(), e['label']) for e in result.get('true_entities', []))
        pred_entities = set((e['text'].lower(), e['label']) for e in result['predicted_entities'])
        
        metrics['synthetic_metrics']['tp'] += len(true_entities.intersection(pred_entities))
        metrics['synthetic_metrics']['fp'] += len(pred_entities - true_entities)
        metrics['synthetic_metrics']['fn'] += len(true_entities - pred_entities)
    
    # Analyze Reddit data
    for result in reddit_results:
        entities_detected = len(result['predicted_entities'])
        if entities_detected > 0:
            metrics['reddit_analysis']['samples_with_pii'] += 1
            metrics['reddit_analysis']['total_entities_detected'] += entities_detected
            
            for entity in result['predicted_entities']:
                entity_type = entity['label']
                if entity_type not in metrics['reddit_analysis']['entities_by_type']:
                    metrics['reddit_analysis']['entities_by_type'][entity_type] = 0
                metrics['reddit_analysis']['entities_by_type'][entity_type] += 1
    
    # Calculate synthetic performance
    tp, fp, fn = metrics['synthetic_metrics']['tp'], metrics['synthetic_metrics']['fp'], metrics['synthetic_metrics']['fn']
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    metrics['synthetic_metrics']['precision'] = precision
    metrics['synthetic_metrics']['recall'] = recall
    metrics['synthetic_metrics']['f1'] = f1
    
    # Calculate Reddit statistics
    if metrics['reddit_analysis']['total_samples'] > 0:
        metrics['reddit_analysis']['pii_prevalence'] = (
            metrics['reddit_analysis']['samples_with_pii'] / 
            metrics['reddit_analysis']['total_samples']
        )
    
    return metrics

# STEP 9: Generate comprehensive results
print("\n" + "="*70)
print("📊 REAL-WORLD EVALUATION RESULTS - REDDIT DATA + SYNTHETIC")
print("="*70)

# Combine results
results = []
for i, (sample, pred) in enumerate(zip(all_test_samples, predictions)):
    result = {
        'text': sample['text'],
        'source': sample.get('source', 'synthetic'),
        'category': sample.get('category', 'reddit'),
        'predicted_entities': pred['predicted_entities']
    }
    
    # Only include true_entities for synthetic samples
    if 'true_entities' in sample:
        result['true_entities'] = sample['true_entities']
    
    results.append(result)

# Calculate metrics
metrics = calculate_metrics_with_reddit(results, len(synthetic_samples))

# Display results
print(f"\n🎯 SYNTHETIC DATA PERFORMANCE (Known Ground Truth):")
print(f"   Precision: {metrics['synthetic_metrics']['precision']:.3f}")
print(f"   Recall:    {metrics['synthetic_metrics']['recall']:.3f}")
print(f"   F1 Score:  {metrics['synthetic_metrics']['f1']:.3f}")
print(f"   True Positives:  {metrics['synthetic_metrics']['tp']}")
print(f"   False Positives: {metrics['synthetic_metrics']['fp']}")
print(f"   False Negatives: {metrics['synthetic_metrics']['fn']}")

print(f"\n📈 REDDIT DATA ANALYSIS (Real-World Patterns):")
reddit_analysis = metrics['reddit_analysis']
print(f"   Total Reddit samples: {reddit_analysis['total_samples']}")
print(f"   Samples with PII detected: {reddit_analysis['samples_with_pii']}")
print(f"   PII prevalence: {reddit_analysis.get('pii_prevalence', 0):.1%}")
print(f"   Total entities detected: {reddit_analysis['total_entities_detected']}")
print(f"   Entity types found: {reddit_analysis['entities_by_type']}")

print(f"\n🔍 DETAILED REDDIT SAMPLE RESULTS:")
print("-" * 70)

# Show interesting Reddit samples
reddit_results = results[len(synthetic_samples):]
interesting_reddit_samples = [r for r in reddit_results if r['predicted_entities']]

for i, result in enumerate(interesting_reddit_samples[:5]):  # Show first 5 interesting ones
    print(f"\n{i+1}. [Reddit - {result['source']}]")
    print(f"   Text: {result['text'][:100]}...")
    print(f"   Detected PII: {[(e['label'], e['text']) for e in result['predicted_entities']]}")

# Show some samples with no PII detected
no_pii_samples = [r for r in reddit_results if not r['predicted_entities']][:3]
if no_pii_samples:
    print(f"\n📝 Samples with No PII Detected:")
    for i, result in enumerate(no_pii_samples):
        print(f"   {i+1}. {result['text'][:80]}...")

print(f"\n🔎 SYNTHETIC SAMPLE PERFORMANCE:")
print("-" * 70)

for i, result in enumerate(results[:len(synthetic_samples)]):
    print(f"\n{i+1}. [Synthetic - {result['category']}]")
    print(f"   Text: {result['text']}")
    print(f"   Expected: {[e['label'] for e in result['true_entities']]}")
    print(f"   Detected: {[e['label'] for e in result['predicted_entities']]}")
    
    true_set = set((e['text'].lower(), e['label']) for e in result['true_entities'])
    pred_set = set((e['text'].lower(), e['label']) for e in result['predicted_entities'])
    
    correct = true_set.intersection(pred_set)
    missed = true_set - pred_set
    extra = pred_set - true_set
    
    if correct:
        print(f"   ✅ Correct: {list(correct)}")
    if missed:
        print(f"   ❌ Missed: {list(missed)}")
    if extra:
        print(f"   ⚠️  False Alarms: {list(extra)}")

# STEP 10: Manual annotation template for Reddit data
def create_reddit_annotation_template(reddit_results, filename="reddit_manual_annotation.json"):
    """Create a template for manual annotation of Reddit samples"""
    annotation_data = []
    
    for i, result in enumerate(reddit_results):
        annotation_entry = {
            'id': i,
            'text': result['text'],
            'source': result['source'],
            'model_predictions': result['predicted_entities'],
            'true_entities': [],  # TO BE FILLED MANUALLY
            'notes': '',
            'annotation_status': 'pending'
        }
        annotation_data.append(annotation_entry)
    
    with open(filename, 'w') as f:
        json.dump(annotation_data, f, indent=2)
    
    print(f"\n📋 Reddit annotation template saved: {filename}")
    print("   Next: Manually annotate the 'true_entities' for Reddit samples")
    return filename

# Create annotation template
annotation_file = create_reddit_annotation_template(reddit_results[:10])  # First 10 for manual annotation

# STEP 11: Save comprehensive report
def save_complete_report(results, metrics, filename="stage3_complete_evaluation.json"):
    """Save complete evaluation report"""
    report = {
        'evaluation_info': {
            'date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
            'model_source': model_path,
            'total_samples': len(results),
            'synthetic_samples': len(synthetic_samples),
            'reddit_samples': len(reddit_results),
            'model_type': 'RoBERTa-base (Fine-tuned for PII)'
        },
        'performance_metrics': metrics,
        'synthetic_results': results[:len(synthetic_samples)],
        'reddit_results': results[len(synthetic_samples):],
        'annotation_file': annotation_file
    }
    
    with open(filename, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"\n💾 Complete evaluation report saved: {filename}")
    return filename

final_report = save_complete_report(results, metrics)

print(f"\n🎉 REAL-WORLD EVALUATION COMPLETED!")
print(f"📁 Model used: {model_path}")
print(f"📊 Report saved: {final_report}")
print(f"📋 Annotation template: {annotation_file}")
print(f"🔍 Reddit samples analyzed: {len(reddit_results)}")
print(f"🎯 Synthetic samples tested: {len(synthetic_samples)}")
print(f"\n📝 For your Stage 3 report, you now have:")
print(f"   - Quantitative metrics from synthetic data")
print(f"   - Qualitative analysis of real Reddit data")
print(f"   - Manual annotation template for further validation")