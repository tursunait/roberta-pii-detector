import torch
import requests
import time
import json
import pandas as pd
from transformers import AutoTokenizer, AutoModelForTokenClassification
import os
import sys

print("Starting Real-World Evaluation with the trained model")

# Configuration
current_dir = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(current_dir, "..", "model", "my_trained_pii_model")  

# Check the BILOU Labels
from transformers import AutoModelForTokenClassification

# Load your trained model
model = AutoModelForTokenClassification.from_pretrained("./model/my_trained_pii_model")

# Check the labels
print("=" * 60)
print("🏷️ YOUR MODEL'S LABELS:")
print("=" * 60)

print(f"\nTotal number of labels: {len(model.config.id2label)}")

print("\nLabel ID → Label Name:")
for label_id, label_name in sorted(model.config.id2label.items()):
    print(f"  {label_id:2d}: {label_name}")

print("\n" + "=" * 60)

def load_model_and_labels(model_path):
    """Load model and extract label mappings"""
    try:
        print(f"Loading model from: {model_path}")
        
        # Check if the model directory exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model directory not found: {model_path}")
        
        print(f"Model directory contents: {os.listdir(model_path)}")
        
        # Load tokenizer and model with local_files_only=True to ensure it uses local path
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        model = AutoModelForTokenClassification.from_pretrained(model_path, local_files_only=True)
        
        # Extract label mappings from model config
        if hasattr(model.config, 'id2label') and model.config.id2label:
            ID2LABEL = model.config.id2label
            # Ensure keys are integers
            if isinstance(list(ID2LABEL.keys())[0], str):
                ID2LABEL = {int(k): v for k, v in ID2LABEL.items()}
            LABEL2ID = {v: k for k, v in ID2LABEL.items()}
        else:
            # Default PII labels (adjust based on your training)
            print("Using default PII labels")
            default_labels = [
                'O',
                'B-EMAIL', 'I-EMAIL', 'L-EMAIL', 'U-EMAIL',
                'B-PHONE', 'I-PHONE', 'L-PHONE', 'U-PHONE',
                'B-SSN', 'I-SSN', 'L-SSN', 'U-SSN',
                'B-CREDIT_CARD', 'I-CREDIT_CARD', 'L-CREDIT_CARD', 'U-CREDIT_CARD',
                'B-PERSON', 'I-PERSON', 'L-PERSON', 'U-PERSON',
                'B-ORG', 'I-ORG', 'L-ORG', 'U-ORG',
                'B-ADDRESS', 'I-ADDRESS', 'L-ADDRESS', 'U-ADDRESS',
                'B-DATE', 'I-DATE', 'L-DATE', 'U-DATE'
            ]
            ID2LABEL = {i: label for i, label in enumerate(default_labels)}
            LABEL2ID = {v: k for k, v in ID2LABEL.items()}
        
        print(f"Loaded model with {len(ID2LABEL)} labels")
        return model, tokenizer, ID2LABEL, LABEL2ID
        
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please ensure:")
        print("   1. Your model is saved at the correct path")
        print("   2. The model was properly trained and saved")
        print("   3. You have all required packages installed")
        
        # Let's check what's actually in the directory
        if os.path.exists(model_path):
            print(f"Contents of {model_path}:")
            for file in os.listdir(model_path):
                print(f"   - {file}")
        else:
            print(f"Directory {model_path} does not exist")
            
        raise

# Load the model
model, tokenizer, ID2LABEL, LABEL2ID = load_model_and_labels(MODEL_PATH)

# STEP 1: Collect Real Reddit Data
def collect_reddit_samples(subreddits=['relationships', 'personalfinance', 'jobs', 'askreddit'], limit=15):
    """
    Collect real text samples from Reddit where people might share personal info
    """
    samples = []
    
    for subreddit in subreddits:
        try:
            print(f"Fetching from r/{subreddit}...")
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
                print(f"Could not fetch from r/{subreddit}: Status {response.status_code}")
                
            time.sleep(1)  
            
        except Exception as e:
            print(f"Error fetching from r/{subreddit}: {e}")
    
    print(f"Collected {len(samples)} samples from Reddit")
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
        },
        {
            'text': "Please ship to 123 Maple Street, Springfield IL 62704 and bill my card 4111-1111-1111-1111",
            'true_entities': [
                {'start': 15, 'end': 56, 'label': 'ADDRESS', 'text': '123 Maple Street, Springfield IL 62704'},
                {'start': 74, 'end': 93, 'label': 'CREDIT_CARD', 'text': '4111-1111-1111-1111'}
            ],
            'source': 'synthetic_shipping',
            'category': 'shipping_billing'
        },
        {
            'text': "My name is Sarah Johnson and I work at Google as a software engineer",
            'true_entities': [
                {'start': 11, 'end': 24, 'label': 'PERSON', 'text': 'Sarah Johnson'},
                {'start': 41, 'end': 47, 'label': 'ORG', 'text': 'Google'}
            ],
            'source': 'synthetic_employment',
            'category': 'employment'
        },
        {
            'text': "The weather is nice today and I went for a walk in the park",
            'true_entities': [],
            'source': 'synthetic_no_pii',
            'category': 'no_pii'
        }
    ]

# STEP 3: Advanced prediction function (FIXED VERSION)
def predict_entities_advanced(texts, model, tokenizer, id2label):
    """Use your trained model for entity prediction with proper token alignment"""
    all_predictions = []
    
    for text in texts:
        try:
            # Tokenize with return_offsets_mapping to get character positions
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, 
                              return_offsets_mapping=True, padding=False)

            
            # Get offset mapping (character positions for each token)
            offset_mapping = inputs.pop('offset_mapping')[0].cpu().numpy()  # Convert to numpy array
            
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
                        'start': int(start_char),  # Convert to Python int
                        'end': int(end_char)       # Convert to Python int
                    }
                elif label.startswith('I-') and current_entity and current_entity['label'] == label[2:]:
                    # Continue entity - extend the text and end position
                    current_entity['text'] += text[start_char:end_char]
                    current_entity['end'] = int(end_char)  # Convert to Python int
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
            print(f"Error processing text: {e}")
            all_predictions.append({
                'text': text,
                'predicted_entities': [],
                'tokens': [],
                'labels': []
            })
    
    return all_predictions

# STEP 4: Calculate metrics
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

# Calculate metrics with manual annotations
def calculate_reddit_metrics_with_manual_annotation(annotation_file="reddit_manual_annotations_COMPLETED.json"):
    """Calculate performance metrics using manually annotated Reddit data"""
    try:
        with open(annotation_file, 'r') as f:
            annotated_data = json.load(f)
        
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        
        completed_samples = [sample for sample in annotated_data if sample.get('annotation_status') == 'completed']
        
        print(f"Calculating metrics on {len(completed_samples)} manually annotated Reddit samples...")
        
        for sample in completed_samples:
            true_entities = set((e['text'].lower(), e['label']) for e in sample.get('true_entities', []))
            pred_entities = set((e['text'].lower(), e['label']) for e in sample.get('model_predictions', []))
            
            true_positives += len(true_entities.intersection(pred_entities))
            false_positives += len(pred_entities - true_entities)
            false_negatives += len(true_entities - pred_entities)
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'true_positives': true_positives,
            'false_positives': false_positives,
            'false_negatives': false_negatives,
            'total_annotated_samples': len(completed_samples)
        }
        
    except FileNotFoundError:
        print(f"Annotation file {annotation_file} not found.")
        return None

# STEP 5: Manual annotation template 
def create_reddit_annotation_template(reddit_results, filename="reddit_manual_annotation.json"):
    """Create a template for manual annotation of Reddit samples"""
    annotation_data = []
    
    for i, result in enumerate(reddit_results):
        # Clean the predicted entities to remove any Tensor objects
        clean_predicted_entities = []
        for entity in result['predicted_entities']:
            # Convert any Tensor objects to Python native types
            clean_entity = {}
            for key, value in entity.items():
                if hasattr(value, 'item'):  # Check if it's a Tensor
                    clean_entity[key] = value.item()  # Convert to Python scalar
                elif hasattr(value, 'tolist'):  # Check if it's a numpy array
                    clean_entity[key] = value.tolist()
                else:
                    clean_entity[key] = value
            clean_predicted_entities.append(clean_entity)
        
        annotation_entry = {
            'id': i,
            'text': result['text'],
            'source': result['source'],
            'model_predictions': clean_predicted_entities,
            'true_entities': [],  # TO BE FILLED MANUALLY
            'notes': '',
            'annotation_status': 'pending'
        }
        annotation_data.append(annotation_entry)
    
    with open(filename, 'w') as f:
        json.dump(annotation_data, f, indent=2, default=str)  # Added default=str for safety
    
    print(f"Reddit annotation template saved: {filename}")
    return filename

# STEP 6: Save comprehensive report (IMPROVED VERSION)
def save_complete_report(results, metrics, annotation_file, filename="stage3_complete_evaluation.json"):
    """Save complete evaluation report"""
    
    # Clean results for JSON serialization
    clean_results = []
    for result in results:
        clean_result = {
            'text': result['text'],
            'source': result['source'],
            'category': result['category'],
            'predicted_entities': []
        }
        
        # Clean predicted entities - ensure all values are JSON serializable
        for entity in result['predicted_entities']:
            clean_entity = {}
            for key, value in entity.items():
                # Convert any tensor-like objects to basic Python types
                if hasattr(value, 'item'):  # PyTorch Tensor
                    clean_entity[key] = value.item()
                elif hasattr(value, 'tolist'):  # Numpy array
                    clean_entity[key] = value.tolist()
                else:
                    clean_entity[key] = value
            clean_result['predicted_entities'].append(clean_entity)
        
        # Add true_entities if they exist (for synthetic samples)
        if 'true_entities' in result:
            clean_result['true_entities'] = result['true_entities']
        
        clean_results.append(clean_result)
    
    report = {
        'evaluation_info': {
            'date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
            'model_source': MODEL_PATH,
            'total_samples': len(clean_results),
            'synthetic_samples': len(synthetic_samples),
            'reddit_samples': len(clean_results) - len(synthetic_samples),
            'model_type': 'RoBERTa-base (Fine-tuned for PII)'
        },
        'performance_metrics': metrics,
        'synthetic_results': clean_results[:len(synthetic_samples)],
        'reddit_results': clean_results[len(synthetic_samples):],
        'annotation_file': annotation_file
    }
    
    with open(filename, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"Complete evaluation report saved: {filename}")
    return filename

# STEP 7: Latency measurement
def measure_latency(texts, model, tokenizer, num_runs=5):
    """Measure inference latency"""
    print("\n Measuring inference latency...")
    latencies = []
    
    # Use shorter texts for latency measurement
    test_texts = texts[:min(5, len(texts))]
    
    for text in test_texts:
        start_time = time.time()
        
        # Tokenize and predict
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
        with torch.no_grad():
            outputs = model(**inputs)
        
        end_time = time.time()
        latency = (end_time - start_time) * 1000  # Convert to milliseconds
        latencies.append(latency)
    
    avg_latency = sum(latencies) / len(latencies)
    max_latency = max(latencies)
    min_latency = min(latencies)
    
    print(f"   Average: {avg_latency:.2f} ms")
    print(f"   Min: {min_latency:.2f} ms")
    print(f"   Max: {max_latency:.2f} ms")
    print(f"   Text length: {len(texts[0])} characters")
    
    return {
        'average': avg_latency,
        'min': min_latency,
        'max': max_latency,
        'all_latencies': latencies
    }

# MAIN EXECUTION
print("\n" + "="*70)
print("REAL-WORLD EVALUATION PIPELINE")
print("="*70)

# Collect data
print("\n Step 1: Collecting real-world data from Reddit...")
reddit_samples = collect_reddit_samples(limit=10)  # Reduced for faster testing

print("\n Step 2: Creating synthetic test samples...")
synthetic_samples = create_synthetic_test_samples()

# Combine all test samples
all_test_samples = synthetic_samples + reddit_samples[:15]  # Use first 15 Reddit samples

print(f"\n Dataset Composition:")
print(f"   - Synthetic samples: {len(synthetic_samples)}")
print(f"   - Reddit samples: {len(reddit_samples[:15])}")
print(f"   - Total samples: {len(all_test_samples)}")

# Run predictions
print("\n Step 3: Running model predictions...")
texts = [sample['text'] for sample in all_test_samples]
predictions = predict_entities_advanced(texts, model, tokenizer, ID2LABEL)

# Measure latency
latency_results = measure_latency(texts, model, tokenizer)

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
print("\n Step 4: Calculating performance metrics...")
metrics = calculate_metrics_with_reddit(results, len(synthetic_samples))

# USE EXISTING MANUAL ANNOTATIONS INSTEAD OF CREATING NEW TEMPLATE
print("\n Step 5: Loading manual annotations...")
completed_annotation_file = "reddit_manual_annotations_COMPLETED.json"

if os.path.exists(completed_annotation_file):
    print(f"Using existing manual annotations: {completed_annotation_file}")
    annotation_file = completed_annotation_file
else:
    print(f"No completed manual annotations found.")
    # Only create template if no completed annotations exist
    template_file = create_reddit_annotation_template(results[len(synthetic_samples):][:10])
    print(f"Manual annotation template created: {template_file}")
    print(f"Please:")
    print(f"   1. Manually annotate {template_file}")
    print(f"   2. Rename it to '{completed_annotation_file}'")
    print(f"   3. Run this script again to use the manual annotations")
    annotation_file = template_file

# Calculate metrics with manual annotations
print("\n Step 6: Calculating metrics with manual annotations...")
manual_metrics = calculate_reddit_metrics_with_manual_annotation(completed_annotation_file)

# Save final report
print("\n Step 7: Generating final report...")
final_report = save_complete_report(results, metrics, annotation_file)

# STEP 8: Display comprehensive results
print("\n" + "="*70)
print(" EVALUATION RESULTS SUMMARY")
print("="*70)

print(f"\n SYNTHETIC DATA PERFORMANCE (Known Ground Truth):")
synth_metrics = metrics['synthetic_metrics']
print(f"   Precision: {synth_metrics['precision']:.3f}")
print(f"   Recall:    {synth_metrics['recall']:.3f}")
print(f"   F1 Score:  {synth_metrics['f1']:.3f}")
print(f"   True Positives:  {synth_metrics['tp']}")
print(f"   False Positives: {synth_metrics['fp']}")
print(f"   False Negatives: {synth_metrics['fn']}")

print(f"\n REDDIT DATA ANALYSIS (Real-World Patterns):")
reddit_analysis = metrics['reddit_analysis']
print(f"   Total Reddit samples: {reddit_analysis['total_samples']}")
print(f"   Samples with PII detected: {reddit_analysis['samples_with_pii']}")
print(f"   PII prevalence: {reddit_analysis.get('pii_prevalence', 0):.1%}")
print(f"   Total entities detected: {reddit_analysis['total_entities_detected']}")
print(f"   Entity types found: {reddit_analysis['entities_by_type']}")

# Display manual annotation metrics if available
if manual_metrics:
    print(f"\n MANUAL ANNOTATION METRICS (Real Ground Truth):")
    print(f"   Precision: {manual_metrics['precision']:.3f}")
    print(f"   Recall:    {manual_metrics['recall']:.3f}")
    print(f"   F1 Score:  {manual_metrics['f1']:.3f}")
    print(f"   True Positives:  {manual_metrics['true_positives']}")
    print(f"   False Positives: {manual_metrics['false_positives']}")
    print(f"   False Negatives: {manual_metrics['false_negatives']}")
    print(f"   Annotated Samples: {manual_metrics['total_annotated_samples']}")

print(f"\n PERFORMANCE CHARACTERISTICS:")
print(f"   Average latency: {latency_results['average']:.2f} ms")
print(f"   Latency range: {latency_results['min']:.2f} - {latency_results['max']:.2f} ms")

print(f"\n DETAILED SAMPLE ANALYSIS:")
print("-" * 70)

# Show synthetic sample results
print(f"\n SYNTHETIC SAMPLES (Ground Truth):")
for i, result in enumerate(results[:len(synthetic_samples)]):
    print(f"\n{i+1}. [{result['category']}]")
    print(f"   Text: {result['text']}")
    print(f"   Expected: {[e['label'] for e in result['true_entities']]}")
    print(f"   Detected: {[e['label'] for e in result['predicted_entities']]}")
    
    true_set = set((e['text'].lower(), e['label']) for e in result['true_entities'])
    pred_set = set((e['text'].lower(), e['label']) for e in result['predicted_entities'])
    
    correct = true_set.intersection(pred_set)
    missed = true_set - pred_set
    extra = pred_set - true_set
    
    if correct:
        print(f"   Correct: {list(correct)}")
    if missed:
        print(f"   Missed: {list(missed)}")
    if extra:
        print(f"   False Alarms: {list(extra)}")

# Show interesting Reddit samples
print(f"\n REDDIT SAMPLES (Real-World Data):")
reddit_results = results[len(synthetic_samples):]
interesting_reddit_samples = [r for r in reddit_results if r['predicted_entities']]

if interesting_reddit_samples:
    print(f"\n Samples with PII Detected:")
    for i, result in enumerate(interesting_reddit_samples[:3]):  # Show first 3
        print(f"\n{i+1}. [Reddit - {result['source']}]")
        print(f"   Text: {result['text'][:120]}...")
        print(f"   Detected PII: {[(e['label'], e['text']) for e in result['predicted_entities']]}")

# Show samples with no PII detected
no_pii_samples = [r for r in reddit_results if not r['predicted_entities']][:2]
if no_pii_samples:
    print(f"\n Samples with No PII Detected:")
    for i, result in enumerate(no_pii_samples):
        print(f"   {i+1}. {result['text'][:80]}...")

print(f"\n" + "="*70)
print(" REAL-WORLD EVALUATION COMPLETED!")
print("="*70)

print(f"\n Output Files:")
print(f"   Evaluation Report: {final_report}")
print(f"   Annotation Template: {annotation_file}")

print(f"\n Stage 3 Report:")
print(f"   Quantitative metrics from {len(synthetic_samples)} synthetic samples")
print(f"   Qualitative analysis of {len(reddit_results)} Reddit samples") 
print(f"   Performance metrics (Precision: {synth_metrics['precision']:.3f}, Recall: {synth_metrics['recall']:.3f}, F1: {synth_metrics['f1']:.3f})")

if manual_metrics:
    print(f"   Manual annotation metrics (Precision: {manual_metrics['precision']:.3f}, Recall: {manual_metrics['recall']:.3f}, F1: {manual_metrics['f1']:.3f})")



print(f"\n Evaluation pipeline finished successfully!")

