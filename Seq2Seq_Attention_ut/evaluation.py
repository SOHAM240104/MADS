import torch
import numpy as np
from sklearn.metrics import classification_report, precision_recall_fscore_support
from collections import defaultdict
import config

def calculate_exact_match(predictions, targets, trg_vocab):
    """
    Calculate Exact Match (EM) score
    Args:
        predictions: [batch_size, seq_len] - predicted token indices
        targets: [batch_size, seq_len] - target token indices
        trg_vocab: vocabulary object
    Returns:
        em_score: float - exact match score
        matches: list of bool - whether each prediction matches exactly
    """
    matches = []
    
    for pred, target in zip(predictions, targets):
        # Convert indices to tokens
        pred_tokens = [trg_vocab.idx2word[idx.item()] for idx in pred]
        target_tokens = [trg_vocab.idx2word[idx.item()] for idx in target]
        
        # Remove padding, sos, and eos tokens for comparison
        pred_clean = [token for token in pred_tokens if token not in ['<pad>', '<sos>', '<eos>']]
        target_clean = [token for token in target_tokens if token not in ['<pad>', '<sos>', '<eos>']]
        
        # Check exact match
        match = pred_clean == target_clean
        matches.append(match)
    
    em_score = sum(matches) / len(matches) if matches else 0.0
    return em_score, matches

def calculate_bleu_score(predictions, targets, trg_vocab):
    """
    Calculate BLEU score (simplified version)
    """
    try:
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        
        bleu_scores = []
        smoothie = SmoothingFunction().method4
        
        for pred, target in zip(predictions, targets):
            # Convert indices to tokens
            pred_tokens = [trg_vocab.idx2word[idx.item()] for idx in pred]
            target_tokens = [trg_vocab.idx2word[idx.item()] for idx in target]
            
            # Remove special tokens
            pred_clean = [token for token in pred_tokens if token not in ['<pad>', '<sos>', '<eos>']]
            target_clean = [token for token in target_tokens if token not in ['<pad>', '<sos>', '<eos>']]
            
            if len(pred_clean) == 0:
                bleu_scores.append(0.0)
            else:
                score = sentence_bleu([target_clean], pred_clean, smoothing_function=smoothie)
                bleu_scores.append(score)
        
        return sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0.0
    
    except ImportError:
        print("NLTK not available for BLEU score calculation")
        return 0.0

def evaluate_by_command_type(predictions, targets, types, trg_vocab):
    """
    Evaluate performance by command type (bash vs docker)
    Args:
        predictions: list of predicted sequences
        targets: list of target sequences  
        types: list of command types ('bash' or 'docker')
        trg_vocab: vocabulary object
    Returns:
        results: dict with per-type metrics
    """
    bash_preds, bash_targets = [], []
    docker_preds, docker_targets = [], []
    
    # Separate by type
    for pred, target, cmd_type in zip(predictions, targets, types):
        if cmd_type == 'bash':
            bash_preds.append(pred)
            bash_targets.append(target)
        elif cmd_type == 'docker':
            docker_preds.append(pred)
            docker_targets.append(target)
    
    results = {}
    
    # Evaluate bash commands
    if bash_preds:
        bash_em, bash_matches = calculate_exact_match(bash_preds, bash_targets, trg_vocab)
        bash_bleu = calculate_bleu_score(bash_preds, bash_targets, trg_vocab)
        results['bash'] = {
            'count': len(bash_preds),
            'exact_match': bash_em,
            'bleu': bash_bleu,
            'matches': bash_matches
        }
    
    # Evaluate docker commands
    if docker_preds:
        docker_em, docker_matches = calculate_exact_match(docker_preds, docker_targets, trg_vocab)
        docker_bleu = calculate_bleu_score(docker_preds, docker_targets, trg_vocab)
        results['docker'] = {
            'count': len(docker_preds),
            'exact_match': docker_em,
            'bleu': docker_bleu,
            'matches': docker_matches
        }
    
    return results

def evaluate_model(model, data_loader, trg_vocab, device, max_len=70):
    """
    Comprehensive model evaluation
    Args:
        model: trained model
        data_loader: DataLoader for evaluation data
        trg_vocab: target vocabulary
        device: torch device
        max_len: maximum sequence length for generation
    Returns:
        evaluation_results: dict with comprehensive metrics
    """
    model.eval()
    
    all_predictions = []
    all_targets = []
    all_types = []
    total_loss = 0
    criterion = torch.nn.CrossEntropyLoss(ignore_index=0, reduction='mean')
    
    with torch.no_grad():
        for batch in data_loader:
            src = batch['src'].to(device)
            trg = batch['trg'].to(device)
            types = batch['types']
            src_lengths = batch['src_lengths']
            
            # Forward pass for loss calculation
            outputs, _ = model(src, trg, src_lengths, teacher_forcing_ratio=0)
            
            # Calculate loss
            outputs_flat = outputs[1:].reshape(-1, outputs.shape[-1])
            trg_flat = trg[1:].reshape(-1)
            loss = criterion(outputs_flat, trg_flat)
            total_loss += loss.item()
            
            # Generate predictions without teacher forcing
            sos_token = trg_vocab.word2idx['<sos>']
            eos_token = trg_vocab.word2idx['<eos>']
            
            predictions, _ = model.inference(src, src_lengths, max_len, sos_token, eos_token)
            
            # Collect results
            batch_size = src.size(1)
            for i in range(batch_size):
                # Get prediction and target for this example
                if predictions.size(0) > 0:
                    pred_seq = predictions[:, i]
                else:
                    pred_seq = torch.tensor([eos_token], device=device)
                
                target_seq = trg[1:, i]  # Skip <sos> token
                
                all_predictions.append(pred_seq)
                all_targets.append(target_seq)
                all_types.append(types[i])
    
    # Calculate overall metrics
    avg_loss = total_loss / len(data_loader)
    overall_em, overall_matches = calculate_exact_match(all_predictions, all_targets, trg_vocab)
    overall_bleu = calculate_bleu_score(all_predictions, all_targets, trg_vocab)
    
    # Calculate per-type metrics
    type_results = evaluate_by_command_type(all_predictions, all_targets, all_types, trg_vocab)
    
    # Create classification report data
    y_true = ['bash' if t == 'bash' else 'docker' for t in all_types]
    
    # For classification report, we need to classify predictions as bash/docker
    # This is a simplified approach - in practice you might want more sophisticated classification
    y_pred = []
    for pred, target, cmd_type in zip(all_predictions, all_targets, all_types):
        # Simple heuristic: if prediction contains 'docker', classify as docker
        pred_tokens = [trg_vocab.idx2word[idx.item()] for idx in pred]
        pred_text = ' '.join(pred_tokens)
        
        if 'docker' in pred_text.lower():
            y_pred.append('docker')
        else:
            y_pred.append('bash')
    
    # Generate classification report
    try:
        class_report = classification_report(y_true, y_pred, 
                                           labels=['bash', 'docker'],
                                           target_names=['bash', 'docker'],
                                           output_dict=True,
                                           zero_division=0)
    except:
        class_report = {"bash": {"precision": 0, "recall": 0, "f1-score": 0},
                       "docker": {"precision": 0, "recall": 0, "f1-score": 0}}
    
    # Compile results
    evaluation_results = {
        'overall': {
            'loss': avg_loss,
            'exact_match': overall_em,
            'bleu': overall_bleu,
            'total_examples': len(all_predictions)
        },
        'by_type': type_results,
        'classification_report': class_report,
        'sample_predictions': get_sample_predictions(all_predictions[:5], all_targets[:5], 
                                                   all_types[:5], trg_vocab)
    }
    
    return evaluation_results

def get_sample_predictions(predictions, targets, types, trg_vocab, num_samples=5):
    """Get sample predictions for inspection"""
    samples = []
    
    for i, (pred, target, cmd_type) in enumerate(zip(predictions[:num_samples], 
                                                   targets[:num_samples], 
                                                   types[:num_samples])):
        # Convert to text
        pred_tokens = [trg_vocab.idx2word[idx.item()] for idx in pred]
        target_tokens = [trg_vocab.idx2word[idx.item()] for idx in target]
        
        # Clean tokens
        pred_clean = [token for token in pred_tokens if token not in ['<pad>', '<sos>', '<eos>']]
        target_clean = [token for token in target_tokens if token not in ['<pad>', '<sos>', '<eos>']]
        
        pred_text = ' '.join(pred_clean)
        target_text = ' '.join(target_clean)
        
        samples.append({
            'type': cmd_type,
            'prediction': pred_text,
            'target': target_text,
            'match': pred_clean == target_clean
        })
    
    return samples

def print_evaluation_results(results):
    """Print evaluation results in a nice format"""
    print("=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)
    
    # Overall metrics
    overall = results['overall']
    print(f"\nOVERALL METRICS:")
    print(f"  Total Examples: {overall['total_examples']}")
    print(f"  Average Loss: {overall['loss']:.4f}")
    print(f"  Exact Match Score: {overall['exact_match']:.4f} ({overall['exact_match']*100:.2f}%)")
    print(f"  BLEU Score: {overall['bleu']:.4f}")
    
    # Per-type metrics
    print(f"\nPER-TYPE METRICS:")
    by_type = results['by_type']
    
    if 'bash' in by_type:
        bash_results = by_type['bash']
        print(f"  BASH Commands ({bash_results['count']} examples):")
        print(f"    Exact Match: {bash_results['exact_match']:.4f} ({bash_results['exact_match']*100:.2f}%)")
        print(f"    BLEU Score: {bash_results['bleu']:.4f}")
    
    if 'docker' in by_type:
        docker_results = by_type['docker']
        print(f"  DOCKER Commands ({docker_results['count']} examples):")
        print(f"    Exact Match: {docker_results['exact_match']:.4f} ({docker_results['exact_match']*100:.2f}%)")
        print(f"    BLEU Score: {docker_results['bleu']:.4f}")
    
    # Classification report
    print(f"\nCLASSIFICATION REPORT:")
    class_report = results['classification_report']
    
    if 'bash' in class_report:
        bash_metrics = class_report['bash']
        print(f"  BASH:")
        print(f"    Precision: {bash_metrics['precision']:.4f}")
        print(f"    Recall: {bash_metrics['recall']:.4f}")
        print(f"    F1-Score: {bash_metrics['f1-score']:.4f}")
    
    if 'docker' in class_report:
        docker_metrics = class_report['docker']
        print(f"  DOCKER:")
        print(f"    Precision: {docker_metrics['precision']:.4f}")
        print(f"    Recall: {docker_metrics['recall']:.4f}")
        print(f"    F1-Score: {docker_metrics['f1-score']:.4f}")
    
    # Sample predictions
    print(f"\nSAMPLE PREDICTIONS:")
    samples = results['sample_predictions']
    for i, sample in enumerate(samples):
        print(f"  Example {i+1} ({sample['type']}):")
        print(f"    Target:     {sample['target']}")
        print(f"    Prediction: {sample['prediction']}")
        print(f"    Match: {'✓' if sample['match'] else '✗'}")
        print()
    
    print("=" * 80)