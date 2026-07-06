import os
import argparse
import pandas as pd


# ---------------------------------------------------------------------------
# Average metric
# ---------------------------------------------------------------------------

def calc_average_metric(results, save_dir, metric):
    average_metric = sum(item[metric] for item in results) / len(results)
    print(f'#Samples: {len(results)}')
    print(f'Average {metric}: {average_metric:.2f}')
    print(f'save_dir: {save_dir}')
    return average_metric


# ---------------------------------------------------------------------------
# OVOBench task-specific accuracy
# ---------------------------------------------------------------------------

OVO_PERCEPTION_TASKS = {
    'ACR': 'Action Recognition',
    'ATR': 'Attribute Recognition (AR)',
    'OJR': 'Object Recognition (OR)',
    'STU': 'Spatial Understanding (SU)',
    'OCR': 'OCR',
    'FPD': 'Future Prediction (FP)',
}

OVO_TRACING_TASKS = {
    'EPM': 'Episodic Memory',
    'HLD': 'Hallucination Detection',
    'ASI': 'Action Sequence Inference',
}

OVO_ALL_TASKS = {**OVO_PERCEPTION_TASKS, **OVO_TRACING_TASKS}

OVO_CATEGORIES = [
    ("Real-Time Visual Perception", OVO_PERCEPTION_TASKS),
    ("Backward Tracing", OVO_TRACING_TASKS),
]

# ---------------------------------------------------------------------------
# StreamingBench task-specific accuracy
# ---------------------------------------------------------------------------

SB_REALTIME_TASKS = [
    'Action Recognition',
    'Attribute Recognition',
    'Object Recognition',
    'Spatial Understanding',
    'Event Understanding',
    'Text-Rich Understanding',
    'Counting',
    'Causal Reasoning',
    'Clips Summarize',
    'Prospective Reasoning',
]

SB_ALL_TASKS = SB_REALTIME_TASKS

# ---------------------------------------------------------------------------
# Unified task-specific accuracy
# ---------------------------------------------------------------------------

def _detect_benchmark(df):
    """Auto-detect benchmark type from task column values."""
    unique_tasks = set(df['task'].dropna().unique())
    ovo_overlap = unique_tasks & set(OVO_ALL_TASKS.keys())
    sb_overlap = unique_tasks & set(SB_ALL_TASKS)
    if len(ovo_overlap) > len(sb_overlap):
        return 'ovobench'
    if len(sb_overlap) > 0:
        return 'streamingbench'
    return 'unknown'


def calc_task_specific_accuracy(df):
    """Calculate accuracy for each task and major category.

    Auto-detects OVOBench vs StreamingBench from the task values.
    Returns (benchmark, task_accuracies, category_results).
    """
    benchmark = _detect_benchmark(df)

    if benchmark == 'ovobench':
        task_list = list(OVO_ALL_TASKS.keys())
        task_display = OVO_ALL_TASKS
        categories = OVO_CATEGORIES
        title = "OVOBench"
    elif benchmark == 'streamingbench':
        task_list = SB_ALL_TASKS
        task_display = {t: t for t in SB_ALL_TASKS}
        categories = []
        title = "StreamingBench"
    else:
        task_list = sorted(df['task'].dropna().unique())
        task_display = {t: t for t in task_list}
        categories = []
        title = "General"

    print("\n" + "=" * 60)
    print(f"TASK-SPECIFIC ACCURACY ({title})")
    print("=" * 60)

    task_accuracies = {}
    for task_key in task_list:
        task_data = df[df['task'] == task_key]
        if len(task_data) > 0:
            acc = task_data['qa_acc'].mean()
            task_accuracies[task_key] = acc
            display = task_display[task_key]
            if display != task_key:
                print(f"{task_key} ({display}): {acc:.2f}% (n={len(task_data)})")
            else:
                print(f"{task_key}: {acc:.2f}% (n={len(task_data)})")

    category_results = []
    if categories:
        print("\n" + "=" * 60)
        print("MAJOR CATEGORY ACCURACY")
        print("=" * 60)

        for cat_name, task_map in categories:
            if isinstance(task_map, dict):
                keys = list(task_map.keys())
            else:
                keys = list(task_map)
            accs = [task_accuracies[k] for k in keys if k in task_accuracies]
            if accs:
                cat_data = df[df['task'].isin(keys)]
                cat_acc = sum(accs) / len(accs)
                category_results.append((cat_name, keys, cat_acc, len(cat_data)))
                print(f"{cat_name}: {cat_acc:.2f}% (n={len(cat_data)})")
                print(f"  - Includes: {', '.join(keys)}")

    print("=" * 60 + "\n")
    return benchmark, task_accuracies, category_results


# ---------------------------------------------------------------------------
# Prediction-choice error analysis
# ---------------------------------------------------------------------------

def analyze_pred_choice_errors(df, debug=False):
    """Count and optionally print rows with invalid pred_choice values.

    Returns error_rate (float) for external file writing.
    """
    if 'pred_choice' not in df.columns:
        return None
    valid_choices = set('ABCDEFGH')
    n_errors = 0
    for _, row in df.iterrows():
        pred_answer = row['pred_answer']
        pred_choice = row['pred_choice']
        if (isinstance(pred_answer, float)
                or len(str(pred_answer)) == 0
                or str(pred_choice)[0] not in valid_choices):
            n_errors += 1
            if debug:
                print(f'Video: {row["video_id"]}, Question: {row["question"]}, '
                      f'GT: {row["correct_choice"]}, Pred: {pred_choice}')
    error_rate = n_errors / len(df) * 100
    print(f'%Errors: {error_rate:.2f}')
    return error_rate


# ---------------------------------------------------------------------------
# Write evaluation results to file
# ---------------------------------------------------------------------------

def write_evaluation_report(save_dir, df, num_samples, average_acc,
                            benchmark=None, task_accuracies=None,
                            category_results=None, error_rate=None):
    """Write a comprehensive evaluation report to a txt file."""
    results_file = os.path.join(save_dir, 'eval_results.txt')

    if benchmark == 'ovobench':
        task_display = OVO_ALL_TASKS
        title = "OVOBench"
    elif benchmark == 'streamingbench':
        task_display = {t: t for t in SB_ALL_TASKS}
        title = "StreamingBench"
    else:
        task_display = ({k: k for k in task_accuracies}
                        if task_accuracies else {})
        title = "General"

    with open(results_file, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write(f"EVALUATION RESULTS ({title})\n")
        f.write("=" * 60 + "\n")
        f.write(f"#Samples: {num_samples}\n")
        f.write(f"Average qa_acc: {average_acc:.2f}\n")
        f.write(f"save_dir: {save_dir}\n")

        if task_accuracies is not None:
            f.write("\n" + "=" * 60 + "\n")
            f.write(f"TASK-SPECIFIC ACCURACY ({title})\n")
            f.write("=" * 60 + "\n")
            for task_key, acc in task_accuracies.items():
                task_data = df[df['task'] == task_key]
                display = task_display.get(task_key, task_key)
                if display != task_key:
                    f.write(f"{task_key} ({display}): {acc:.2f}% "
                            f"(n={len(task_data)})\n")
                else:
                    f.write(f"{task_key}: {acc:.2f}% "
                            f"(n={len(task_data)})\n")

            if category_results:
                f.write("\n" + "=" * 60 + "\n")
                f.write("MAJOR CATEGORY ACCURACY\n")
                f.write("=" * 60 + "\n")
                for cat_name, keys, cat_acc, cat_n in category_results:
                    f.write(f"{cat_name}: {cat_acc:.2f}% (n={cat_n})\n")
                    f.write(f"  - Includes: {', '.join(keys)}\n")
            f.write("=" * 60 + "\n")

        if error_rate is not None:
            f.write(f"\n%Errors: {error_rate:.2f}\n")

    print(f"Evaluation report saved to: {results_file}")


# ---------------------------------------------------------------------------
# EgoSchema submission generation
# ---------------------------------------------------------------------------

def generate_egoschema_submission(df, save_dir):
    """Convert predictions to EgoSchema submission format (q_uid, answer)."""
    records = df.to_dict(orient='records')
    submission = []
    for r in records:
        choice = r.get('pred_choice', 'A')
        if choice not in ['A', 'B', 'C', 'D', 'E']:
            print(f"Invalid pred_choice: {choice}")
            choice = 'A'
        submission.append({
            'q_uid': r['video_id'],
            'answer': ord(choice) - ord('A'),
        })
    out_path = os.path.join(save_dir, 'submission.csv')
    pd.DataFrame(submission).to_csv(out_path, index=False)
    print(f'EgoSchema submission saved to: {out_path}')


# ---------------------------------------------------------------------------
# VideoMME duration-based evaluation
# ---------------------------------------------------------------------------

def eval_videomme_by_duration(df, results_path):
    """Report accuracy by duration using duration_category already in results."""
    if 'duration_category' not in df.columns:
        raise ValueError(
            "Column 'duration_category' not found in results. "
            "Make sure inference records duration_category for each sample."
        )
    df['duration'] = df['duration_category']
    merged = df

    duration_stats = merged.groupby('duration').agg(
        {'qa_acc': ['count', 'mean', 'sum']}
    ).round(2)
    duration_stats.columns = ['Total Questions', 'Average Accuracy (%)',
                              'Total Correct']
    duration_stats['Total Correct'] = \
        (duration_stats['Total Correct'] / 100).astype(int)

    total_questions = len(merged)
    total_correct = (merged['qa_acc'] == 100.0).sum()
    overall_accuracy = (total_correct / total_questions * 100
                        if total_questions > 0 else 0)

    save_dir = os.path.dirname(results_path)
    output_file = os.path.join(save_dir, 'eval_results.txt')

    with open(output_file, 'w', encoding='utf-8') as f:
        def write_line(text):
            print(text)
            f.write(text + '\n')

        write_line("=" * 60)
        write_line("Accuracy Statistics by Video Duration")
        write_line("=" * 60)
        write_line(str(duration_stats))
        write_line("=" * 60)

        write_line("\nOverall Statistics:")
        write_line(f"Total Questions: {total_questions}")
        write_line(f"Total Correct: {total_correct}")
        write_line(f"Overall Accuracy: {overall_accuracy:.2f}%")

        write_line("\nDetailed Statistics:")
        for duration in ['short', 'medium', 'long']:
            if duration in merged['duration'].values:
                subset = merged[merged['duration'] == duration]
                total = len(subset)
                correct = (subset['qa_acc'] == 100.0).sum()
                acc = correct / total * 100 if total > 0 else 0
                write_line(f"{duration.capitalize():8s}: {correct:4d}/{total:4d} "
                           f"correct, Accuracy: {acc:.2f}%")

    print(f"\nResults saved to: {output_file}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def build_parser():
    parser = argparse.ArgumentParser(
        description="Unified evaluation script for multiple-choice benchmarks")

    sub = parser.add_subparsers(dest='command', help='Evaluation mode')

    # --- general: the default multi-metric evaluation ---
    p_gen = sub.add_parser('general',
                           help='Compute average metrics, OVOBench breakdown, '
                                'and error analysis')
    p_gen.add_argument('--results_path', type=str, required=True)
    p_gen.add_argument('--debug', action='store_true')

    # --- egoschema: generate submission CSV ---
    p_ego = sub.add_parser('egoschema',
                           help='Generate EgoSchema submission file')
    p_ego.add_argument('--results_path', type=str, required=True)

    # --- videomme: accuracy by video duration ---
    p_vmme = sub.add_parser('videomme',
                            help='Evaluate VideoMME results by duration')
    p_vmme.add_argument('--results_path', type=str, required=True)
    p_vmme.add_argument('--debug', action='store_true')

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    if args.command == 'general':
        df = pd.read_csv(args.results_path)
        save_dir = os.path.dirname(args.results_path)
        results = df.to_dict(orient='records')
        average_acc = calc_average_metric(results, save_dir, 'qa_acc')

        benchmark, task_accuracies, category_results = None, None, None
        if 'task' in df.columns:
            benchmark, task_accuracies, category_results = \
                calc_task_specific_accuracy(df)

        error_rate = analyze_pred_choice_errors(
            df, debug=args.debug)

        write_evaluation_report(
            save_dir, df, len(results), average_acc,
            benchmark, task_accuracies, category_results, error_rate)

    elif args.command == 'egoschema':
        df = pd.read_csv(args.results_path)
        save_dir = os.path.dirname(args.results_path)
        generate_egoschema_submission(df, save_dir)

    elif args.command == 'videomme':
        df = pd.read_csv(args.results_path)
        eval_videomme_by_duration(df, args.results_path)


if __name__ == '__main__':
    main()
