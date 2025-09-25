#!/usr/bin/env python3
"""
Simple test script to validate model functionality
"""

import sys
import os
sys.path.append('Seq2Seq_Attention_ut')

import torch
import json

def test_data_loading():
    """Test if we can load the datasets"""
    print("Testing data loading...")
    
    # Check if data files exist
    data_files = [
        'data/bash_dataset.json',
        'data/docker_dataset.json'
    ]
    
    for file_path in data_files:
        if os.path.exists(file_path):
            print(f"✓ Found {file_path}")
            # Load a sample
            with open(file_path, 'r') as f:
                data = json.load(f)
                print(f"  - Contains {len(data.get('examples', []))} examples")
                if data.get('examples'):
                    sample = data['examples'][0]
                    print(f"  - Sample query: {sample.get('query', 'N/A')[:50]}...")
                    print(f"  - Sample command: {sample.get('command', 'N/A')}")
        else:
            print(f"✗ Missing {file_path}")

def test_model_files():
    """Test if model files exist"""
    print("\nTesting model files...")
    
    model_files = [
        'best_model_seq2seq_attention.bin',
        'model_seq2seq_attention.bin',
        'Seq2Seq/model_seq2seq_1.bin'
    ]
    
    found_models = []
    for model_file in model_files:
        if os.path.exists(model_file):
            size_mb = os.path.getsize(model_file) / (1024 * 1024)
            print(f"✓ Found {model_file} ({size_mb:.1f} MB)")
            found_models.append(model_file)
        else:
            print(f"✗ Missing {model_file}")
    
    return found_models

def test_simple_import():
    """Test if we can import the model modules"""
    print("\nTesting module imports...")
    
    try:
        from Seq2Seq_Attention_ut import config
        print("✓ Successfully imported config")
        print(f"  - Device: {config.device}")
        print(f"  - Batch size: {config.BATCH_SIZE}")
    except Exception as e:
        print(f"✗ Failed to import config: {e}")
        return False
    
    try:
        from Seq2Seq_Attention_ut import model
        print("✓ Successfully imported model")
    except Exception as e:
        print(f"✗ Failed to import model: {e}")
        return False
    
    try:
        from Seq2Seq_Attention_ut import dataset
        print("✓ Successfully imported dataset")
    except Exception as e:
        print(f"✗ Failed to import dataset: {e}")
        return False
    
    return True

def test_dataset_creation():
    """Test if we can create the dataset"""
    print("\nTesting dataset creation...")
    
    try:
        from Seq2Seq_Attention_ut import dataset
        train, valid, test, SRC, TRG, UT = dataset.create_dataset()
        print("✓ Successfully created dataset")
        print(f"  - Training samples: {len(train.examples)}")
        print(f"  - Validation samples: {len(valid.examples)}")
        print(f"  - Test samples: {len(test.examples)}")
        print(f"  - Source vocabulary size: {len(SRC.vocab)}")
        print(f"  - Target vocabulary size: {len(TRG.vocab)}")
        return True, SRC, TRG
    except Exception as e:
        print(f"✗ Failed to create dataset: {e}")
        return False, None, None

def main():
    print("="*60)
    print("MODEL VALIDATION TEST")
    print("="*60)
    
    # Test 1: Data files
    test_data_loading()
    
    # Test 2: Model files
    found_models = test_model_files()
    
    # Test 3: Module imports
    if not test_simple_import():
        print("\n❌ Critical: Cannot import required modules")
        return 1
    
    # Test 4: Dataset creation
    success, SRC, TRG = test_dataset_creation()
    if not success:
        print("\n❌ Critical: Cannot create dataset")
        return 1
    
    print("\n" + "="*60)
    
    if found_models:
        print("✅ GOOD NEWS: You have trained models available!")
        print(f"   Best model to use: {found_models[0]}")
        print("\nYou can now:")
        print("1. Use the inference.py script I created")
        print("2. Run: python inference.py --interactive")
        print("3. Or train a new model if results aren't satisfactory")
    else:
        print("⚠️  WARNING: No trained models found")
        print("\nYou need to:")
        print("1. Train a model first using Seq2Seq_Attention_ut/train.py")
        print("2. Run: cd Seq2Seq_Attention_ut && python train.py --action train")
    
    print("\nNext steps:")
    print("- If you have models: Test the inference script")
    print("- If no models: Train the Seq2Seq_Attention_ut model first")
    print("- The attention-based model should work better than transformers for your use case")
    
    return 0

if __name__ == "__main__":
    exit(main())