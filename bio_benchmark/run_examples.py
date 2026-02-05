import json
import re
import time
import random
import os
import sys
import yaml
from pathlib import Path
from tqdm import tqdm
from utils import LLMCaller


examples_folder = Path('examples/')
output_folder = Path('results/')


def load_existing_questions(result_paths: list) -> set:
    """
    Load all questions that have already been successfully asked from existing result directories.
    Only counts successful runs (llm_call_time != -1 and non-empty response).
    
    Args:
        result_paths: List of paths to result directories (can be strings or Paths)
    
    Returns:
        Set of question strings that have already been successfully run
    """
    existing_questions = set()
    failed_count = 0
    
    for run_path in result_paths:
        results_path = Path(run_path)
        
        if not results_path.exists():
            print(f"Warning: Path does not exist: {results_path}")
            continue
        
        # Iterate through category subdirectories
        for category_dir in results_path.iterdir():
            if category_dir.is_dir():
                for jsonl_file in category_dir.glob("*.jsonl"):
                    with open(jsonl_file, 'r') as f:
                        for line in f:
                            try:
                                result = json.loads(line.strip())
                                if 'question' in result:
                                    # Only count as existing if it didn't fail
                                    # Failed runs have llm_call_time == -1 or empty response
                                    llm_call_time = result.get('llm_call_time', 0)
                                    response = result.get('final_response', '')
                                    
                                    if llm_call_time == -1 or response == '':
                                        failed_count += 1
                                        continue
                                    
                                    existing_questions.add(result['question'])
                            except json.JSONDecodeError:
                                continue
    
    if failed_count > 0:
        print(f"  Found {failed_count} failed runs that will be re-tried")
    
    return existing_questions


def get_all_jsonl_files():
    """Get a sorted list of all jsonl files across all categories."""
    all_files = []
    categories = sorted([d for d in examples_folder.iterdir() if d.is_dir()])
    for category_dir in categories:
        jsonl_files = sorted(category_dir.glob("*.jsonl"))
        for jsonl_file in jsonl_files:
            all_files.append((category_dir.name, jsonl_file))
    return all_files


def append_jsonl(path, obj):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        f.flush()


def load_examples(jsonl_path: Path) -> list:
    """Load examples from a jsonl file."""
    examples = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            examples.append(json.loads(line.strip()))
    return examples


def get_motif_ratio(example: dict) -> float:
    """
    Compute the motif ratio for an example.
    Motif ratio = length_convergent_block / seq_length
    
    Returns:
        float or None if required fields are missing
    """
    block_size = example.get('length_convergent_block')
    seq_length = example.get('seq_length')
    
    if block_size is None or seq_length is None or seq_length == 0:
        return None
    
    return block_size / seq_length


def load_all_examples_with_motif_ratio(motif_ratio_range: tuple = None) -> list:
    """
    Load all examples from all categories and optionally filter by motif ratio.
    
    Args:
        motif_ratio_range: Tuple of (min_ratio, max_ratio). If provided, only
                          examples within this range are returned.
    
    Returns:
        List of (category_name, example) tuples
    """
    all_examples = []
    
    categories = sorted([d for d in examples_folder.iterdir() if d.is_dir()])
    for category_dir in categories:
        category_name = category_dir.name
        for jsonl_file in sorted(category_dir.glob("*.jsonl")):
            examples = load_examples(jsonl_file)
            for example in examples:
                ratio = get_motif_ratio(example)
                
                # If no filter, include all
                if motif_ratio_range is None:
                    all_examples.append((category_name, example))
                elif ratio is not None:
                    min_ratio, max_ratio = motif_ratio_range
                    if min_ratio <= ratio < max_ratio:
                        all_examples.append((category_name, example))
    
    return all_examples


def parse_llm_response(response: str, num_expected_taxa: int):
    """
    Parse the LLM response to extract yes/no answer and identified taxa.
    
    Returns:
        (said_yes: bool or None, identified_taxa: list of str)
        said_yes is None if parsing failed
    """
    response_lower = response.lower()
    
    # Check for yes/no
    has_yes = 'yes' in response_lower
    has_no = 'no' in response_lower
    
    if has_yes and has_no:
        said_yes = None
    elif has_yes:
        said_yes = True
    elif has_no:
        said_yes = False
    else:
        said_yes = None
    
    # Extract taxa - look for patterns like "taxon_1", "taxon_2", etc.
    taxa_pattern = r'taxon[_\s]?(\d+)'
    identified_taxa = re.findall(taxa_pattern, response_lower)
    
    # If no taxon_X pattern found, try to find standalone numbers
    if not identified_taxa:
        identified_taxa = re.findall(r'\b(\d+)\b', response_lower)
    
    return said_yes, identified_taxa


def calculate_taxa_metrics(ground_truth_taxa: list, predicted_taxa: list) -> dict:
    """
    Calculate precision, recall, and F1 score for taxa identification.
    """
    if not ground_truth_taxa:
        return {'precision': -1, 'recall': -1, 'f1': -1}
    
    # Normalize taxa names for comparison
    gt_set = set(str(t).lower() for t in ground_truth_taxa)
    pred_set = set(str(t).lower() for t in predicted_taxa)
    
    overlap = gt_set.intersection(pred_set)
    
    recall = len(overlap) / len(gt_set)
    
    if len(pred_set) == 0:
        precision = 0.0
    else:
        precision = len(overlap) / len(pred_set)
    
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * (precision * recall) / (precision + recall)
    
    return {'precision': precision, 'recall': recall, 'f1': f1}


def run_examples_on_llm(config, file_index=None, run_id=None, skip_existing_paths=None, motif_ratio_range=None):
    """
    Load pre-generated examples and run them through the LLM.
    
    Args:
        config: Configuration dictionary
        file_index: If provided, only process this file (0-indexed)
        run_id: If provided, use this run_id instead of generating one
        skip_existing_paths: List of result directory paths. Questions already in these
                             directories will be skipped to avoid re-running.
        motif_ratio_range: Tuple of (min_ratio, max_ratio). If provided, only examples
                           with motif ratio in this range will be processed.
    """
    # Load existing questions to skip
    existing_questions = set()
    if skip_existing_paths:
        print(f"Loading existing results to skip from {len(skip_existing_paths)} path(s)...")
        existing_questions = load_existing_questions(skip_existing_paths)
        print(f"  Found {len(existing_questions)} existing questions to skip")
    
    is_random_mode = config.get('backend') == 'random'
    
    if is_random_mode:
        LLM = None
        model_name = 'random_baseline'
    else:
        LLM = LLMCaller(config)
        model_name = config['model'].replace('/', '_')
    
    if run_id is None:
        run_id = f"{model_name}_{int(time.time())}_{random.randint(0, 999999):06d}"
    run_dir = output_folder / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all jsonl files
    all_files = get_all_jsonl_files()
    
    if file_index is not None:
        # Process only the specified file
        if file_index < 0 or file_index >= len(all_files):
            print(f"Error: file_index {file_index} out of range. Valid range: 0-{len(all_files)-1}")
            return
        files_to_process = [all_files[file_index]]
    else:
        files_to_process = all_files

    # files_to_process = [i for i in all_files if 'num_taxa' in i[1].name.lower()]
    
    # Motif ratio exploration mode - process all examples across categories with ratio filter
    if motif_ratio_range is not None:
        print(f"\nMotif Ratio Exploration Mode: filtering to ratio range {motif_ratio_range[0]:.2%} - {motif_ratio_range[1]:.2%}")
        all_filtered_examples = load_all_examples_with_motif_ratio(motif_ratio_range)
        print(f"  Found {len(all_filtered_examples)} examples in range")
        
        if len(all_filtered_examples) == 0:
            print("No examples found in the specified motif ratio range!")
            return
        
        # Group by category for output organization
        examples_by_category = {}
        for cat_name, example in all_filtered_examples:
            if cat_name not in examples_by_category:
                examples_by_category[cat_name] = []
            examples_by_category[cat_name].append(example)
        
        # Process each category
        for category_name, examples in examples_by_category.items():
            print(f"\n{'='*60}")
            print(f"Processing category: {category_name} ({len(examples)} examples in ratio range)")
            print('='*60)
            
            category_output_dir = run_dir / category_name
            category_output_dir.mkdir(parents=True, exist_ok=True)
            
            output_file = category_output_dir / f"motif_ratio_{motif_ratio_range[0]:.2f}_{motif_ratio_range[1]:.2f}.jsonl"
            
            if output_file.exists():
                output_file.unlink()
            
            cleaned_response = ""
            skipped_count = 0
            for example in tqdm(examples, desc=category_name):
                question = example['question']
                
                if question in existing_questions:
                    skipped_count += 1
                    continue
                
                question_modified = question.replace('Return your answer as: Yes/No and if Yes, list the taxa involved.', 'Return your answer as: Yes/No and if Yes, list the taxa involved. Nothing else.')
                ground_truth_taxa = example.get('ground_truth_taxa', [])
                num_taxa = example.get('num_convergent_taxa', len(ground_truth_taxa))
                num_leaves = example.get('num_leaves', 50)
                
                if is_random_mode:
                    num_guesses = random.randint(10, 12)
                    identified_taxa = [str(i) for i in random.sample(range(1, num_leaves + 1), min(num_guesses, num_leaves))]
                    said_yes = True
                    response = f"Yes. Random guess: {', '.join(['taxon_' + t for t in identified_taxa])}"
                    llm_call_time = 0.0
                else:
                    conversation = [{"role": "user", "content": question_modified}]
                    try:
                        response, llm_call_time = LLM.call_openai(
                            conversation,
                            model=config['model'],
                            temp=0.7,
                            max_tokens=2000
                        )
                        if config['backend'] == 'anthropic':
                            cleaned_response = ' '.join([i for i in response.split('\n') if i][-2:])
                    except Exception as e:
                        print(f"LLM call failed: {e}")
                        response = ""
                        cleaned_response = ""
                        llm_call_time = -1
                    
                    if config['backend'] == 'anthropic':
                        said_yes, identified_taxa = parse_llm_response(cleaned_response, num_taxa)
                    else:
                        said_yes, identified_taxa = parse_llm_response(response, num_taxa)
                
                if said_yes is None:
                    metrics = {'precision': -1, 'recall': -1, 'f1': -1}
                elif said_yes:
                    metrics = calculate_taxa_metrics(ground_truth_taxa, identified_taxa)
                else:
                    metrics = {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
                print(metrics)
                
                result = {
                    **example,
                    'final_response': response,
                    'said_yes': said_yes,
                    'identified_taxa': identified_taxa,
                    'precision': metrics['precision'],
                    'recall': metrics['recall'],
                    'f1': metrics['f1'],
                    'model': config.get('model', 'random'),
                    'llm_call_time': llm_call_time,
                    'motif_ratio': get_motif_ratio(example),
                }
                
                append_jsonl(output_file, result)
            
            if skipped_count > 0:
                print(f"  Skipped {skipped_count} already-run questions")
        
        print(f"\n{'='*60}")
        print(f"DONE! Results saved to: {run_dir}")
        print('='*60)
        return
    
    # Normal mode - process files
    
    for category_name, jsonl_file in files_to_process:
        print(f"\n{'='*60}")
        print(f"Processing category: {category_name}, file: {jsonl_file.name}")
        print('='*60)
        
        # Create output directory for this category
        category_output_dir = run_dir / category_name
        category_output_dir.mkdir(parents=True, exist_ok=True)
            
        examples = load_examples(jsonl_file)
        output_file = category_output_dir / jsonl_file.name
        
        # Clear output file if it exists
        if output_file.exists():
            output_file.unlink()
            
        cleaned_response = ""
        skipped_count = 0
        for example in tqdm(examples, desc=jsonl_file.stem):
            question = example['question']
            
            # Skip if this question was already asked
            if question in existing_questions:
                print("SKIPPED!")
                skipped_count += 1
                continue
            
            question = question.replace('Return your answer as: Yes/No and if Yes, list the taxa involved.', 'Return your answer as: Yes/No and if Yes, list the taxa involved. Nothing else.')
            ground_truth_taxa = example.get('ground_truth_taxa', [])
            num_taxa = example.get('num_convergent_taxa', len(ground_truth_taxa))
            num_leaves = example.get('num_leaves', 50)
            
            if is_random_mode:
                # Random baseline: guess 10-12 random taxa
                num_guesses = random.randint(10, 12)
                identified_taxa = [str(i) for i in random.sample(range(1, num_leaves + 1), min(num_guesses, num_leaves))]
                said_yes = True
                response = f"Yes. Random guess: {', '.join(['taxon_' + t for t in identified_taxa])}"
                llm_call_time = 0.0
            else:
                # Call LLM
                conversation = [{"role": "user", "content": question}]
                try:
                    response, llm_call_time = LLM.call_openai(
                        conversation,
                        model=config['model'],
                        temp=0.7,
                        max_tokens=2000
                    )
                    if config['backend'] == 'anthropic':
                        cleaned_response = ' '.join([i for i in response.split('\n') if i][-2:])
                except Exception as e:
                    print(f"LLM call failed: {e}")
                    response = ""
                    cleaned_response = ""
                    llm_call_time = -1
                
                # Parse response
                if config['backend'] == 'anthropic':
                    said_yes, identified_taxa = parse_llm_response(cleaned_response, num_taxa)
                else:
                    said_yes, identified_taxa = parse_llm_response(response, num_taxa)
            
            # Calculate metrics
            if said_yes is None:
                metrics = {'precision': -1, 'recall': -1, 'f1': -1}
            elif said_yes:
                metrics = calculate_taxa_metrics(ground_truth_taxa, identified_taxa)
            else:
                metrics = {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
            print(metrics)
            # Store results (include original example data + LLM results)
            result = {
                **example,  # Include all original example data
                'final_response': response,
                'said_yes': said_yes,
                'identified_taxa': identified_taxa,
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'f1': metrics['f1'],
                'model': config.get('model', 'random'),
                'llm_call_time': llm_call_time,
            }
            
            append_jsonl(output_file, result)
        
        if skipped_count > 0:
            print(f"  Skipped {skipped_count} already-run questions")
    
    print(f"\n{'='*60}")
    print(f"DONE! Results saved to: {run_dir}")
    print('='*60)


if __name__ == "__main__":
    with open('configs/config.yaml', 'r') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    
    # Usage:
    #   python -m phylo.run_examples                    # Run all files
    #   python -m phylo.run_examples 0                  # Run file index 0
    #   python -m phylo.run_examples 0 my_run_id        # Run file 0 with shared run_id
    #   python -m phylo.run_examples --list             # List all files with indices
    #   python -m phylo.run_examples --skip-existing path1 path2 ...  # Skip questions from these paths
    #   python -m phylo.run_examples 0 my_run_id --skip-existing path1 path2
    #   python -m phylo.run_examples --motif-ratio-range 0.1 0.2      # Only examples with 10-20% motif ratio
    #   python -m phylo.run_examples --motif-ratio-range 0.1 0.2 --run-id my_run --skip-existing path1
    
    # Parse arguments
    args = sys.argv[1:]
    
    # Check for --skip-existing flag
    skip_existing_paths = None
    if '--skip-existing' in args:
        skip_idx = args.index('--skip-existing')
        # Find where skip-existing paths end (next flag or end)
        end_idx = len(args)
        for i in range(skip_idx + 1, len(args)):
            if args[i].startswith('--'):
                end_idx = i
                break
        skip_existing_paths = args[skip_idx + 1:end_idx]
        args = args[:skip_idx] + args[end_idx:]
    
    # Check for --motif-ratio-range flag
    motif_ratio_range = None
    if '--motif-ratio-range' in args:
        ratio_idx = args.index('--motif-ratio-range')
        try:
            min_ratio = float(args[ratio_idx + 1])
            max_ratio = float(args[ratio_idx + 2])
            motif_ratio_range = (min_ratio, max_ratio)
            args = args[:ratio_idx] + args[ratio_idx + 3:]
        except (IndexError, ValueError):
            print("Error: --motif-ratio-range requires two float arguments (min max)")
            print("Example: --motif-ratio-range 0.1 0.2")
            sys.exit(1)
    
    # Check for --run-id flag (for use with motif ratio mode)
    run_id = None
    if '--run-id' in args:
        run_id_idx = args.index('--run-id')
        run_id = args[run_id_idx + 1]
        args = args[:run_id_idx] + args[run_id_idx + 2:]
    
    if len(args) > 0:
        if args[0] == '--list':
            all_files = get_all_jsonl_files()
            print(f"Total files: {len(all_files)}")
            for i, (cat, f) in enumerate(all_files):
                print(f"  {i}: {cat}/{f.name}")
        elif args[0] == '--list-motif-ratios':
            # Show distribution of motif ratios across all examples
            all_examples = load_all_examples_with_motif_ratio()
            ratios = [get_motif_ratio(ex) for _, ex in all_examples if get_motif_ratio(ex) is not None]
            print(f"Total examples with motif ratio: {len(ratios)}")
            print(f"  Min: {min(ratios):.4f} ({min(ratios)*100:.2f}%)")
            print(f"  Max: {max(ratios):.4f} ({max(ratios)*100:.2f}%)")
            print(f"  Mean: {sum(ratios)/len(ratios):.4f} ({sum(ratios)/len(ratios)*100:.2f}%)")
            # Show bins
            bins = [(0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5), (0.5, 1.0)]
            print("\nDistribution:")
            for low, high in bins:
                count = sum(1 for r in ratios if low <= r < high)
                print(f"  {low:.0%}-{high:.0%}: {count} examples")
        else:
            file_index = int(args[0])
            run_id = run_id or (args[1] if len(args) > 1 else None)
            run_examples_on_llm(config, file_index=file_index, run_id=run_id, 
                               skip_existing_paths=skip_existing_paths,
                               motif_ratio_range=motif_ratio_range)
    else:
        run_examples_on_llm(config, run_id=run_id or 'claude_num_taxa', 
                           skip_existing_paths=skip_existing_paths,
                           motif_ratio_range=motif_ratio_range)