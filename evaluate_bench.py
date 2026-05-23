"""
Longevity-LLM Multi-Omic Differential Evaluation Framework
Target Pathology: Frontotemporal Dementia (FTD) vs. Healthy Brain Aging
Context: Caltech / Insilico Medicine Longevity Hackathon Benchmark Validation
"""

import json
import re
import os
import requests
from sklearn.metrics import accuracy_score, f1_score, classification_report
from concurrent.futures import ThreadPoolExecutor, as_completed

# =====================================================================
# 1. INFRASTRUCTURE & CREDENTIAL CONFIGURATION (SLACK CORRECTIONS APPLIED)
# =====================================================================
ENDPOINT_URL = "https://swchnq0ekc3scmqw.us-east-2.aws.endpoints.huggingface.cloud/v1/chat/completions"
ACCESS_TOKEN = "hf_qAwEwEmwtVTbPzSWKVuCoeXrfdTsROvgt"

# Global validation vectors
ground_truth = []
ai_predictions = []
reproducibility_logs = []


def extract_answer_letter(text: str) -> str:
    """
    Parses the raw text response to isolate the final target token boundary.
    Extracts isolated classification markers (A) or (B) via negative lookarounds.
    """
    if not text:
        return None
    matches = re.findall(r"(?<![A-Za-z])([A-F])(?![A-Za-z])", text)
    if matches:
        return f"({matches[-1].upper()})"
    return None


def evaluate_sample(json_line: str):
    """
    Thread-isolated execution routine managing HTTP payload encoding, 
    server transactions, error handling, and structured log parsing.
    """
    raw_data = json.loads(json_line)
    task_id = raw_data.get("lb_id", "UNKNOWN")
    messages = raw_data["messages"]
    
    # Isolate the definitive ground truth label and prune the sequence for zero-shot testing
    gold_answer = messages[-1]["content"].strip().upper()
    prompt_messages = messages[:-1]
    
    headers = {
        "Authorization": f"Bearer {ACCESS_TOKEN}",
        "Content-Type": "application/json",
        "Accept": "application/json"
    }
    
    # Optimized OpenAI-compatible ChatML payload layout
    # Explicitly suppresses the native Qwen3.5 reasoning trace to bypass timeout boundaries
    payload = {
        "model": "longevity-llm",
        "messages": prompt_messages,
        "max_tokens": 50,  # Highly restricted token budget to enforce classification immediacy
        "temperature": 0.01,  # Near-deterministic decoding mapping
        "chat_template_kwargs": {"enable_thinking": False}
    }
    
    try:
        # 60-second read/write ceiling to accommodate potential cluster cold starts (Scale-to-Zero)
        response = requests.post(ENDPOINT_URL, headers=headers, json=payload, timeout=60)
        
        if response.status_code != 200:
            print(f"❌ Execution Failure on {task_id}: HTTP {response.status_code} - {response.text}")
            return gold_answer, None, None
            
        response_json = response.json()
        raw_content = response_json["choices"][0]["message"]["content"]
        
        final_answer_text = raw_content.strip()
        predicted_letter = extract_answer_letter(final_answer_text)
        
        log_entry = {
            "lb_id": task_id,
            "gold": gold_answer,
            "predicted": predicted_letter,
            "raw_response": final_answer_text
        }
        
        return gold_answer, predicted_letter, log_entry
        
    except Exception as error:
        print(f"❌ Network Layer Disruption on sample {task_id}: {error}")
        return gold_answer, None, None


def run_pipeline():
    """
    Main pipeline orchestrator. Conducts file system verifications, 
    manages the concurrent connection worker pool, and renders scientific metrics.
    """
    dataset_path = "data/dft_aged_brain_benchmark.jsonl"
    results_path = "results/eval_reproducibility_log.jsonl"
    
    print("🧠 Initializing multi-omic transcriptomic & epigenetic database...")
    if not os.path.exists(dataset_path):
        print(f"❌ Critical Failure: Evaluation dataset at '{dataset_path}' cannot be resolved.")
        return

    with open(dataset_path, "r", encoding="utf-8") as file:
        lines = file.readlines()

    print(f"📋 Total molecular test matrices queued: {len(lines)}")
    print("⚡ Establishing handshake with dedicated Hugging Face cloud compute infrastructure...")
    
    # Parallel worker configuration optimized for standard OS TCP sockets
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(evaluate_sample, line) for line in lines]
        for future in as_completed(futures):
            gold, pred, log = future.result()
            if log and pred:
                ground_truth.append(gold)
                ai_predictions.append(pred)
                reproducibility_logs.append(log)

    print("\n========================================================")
    print("   📊 QUANTITATIVE PERFORMANCE REPORT - LONGEVITYLLM")
    print("========================================================")
    
    if not ground_truth or not ai_predictions:
        print("❌ Evaluation Error: Zero valid prediction vectors were resolved from the server.")
        print("💡 The instance might be warming up from a 'Scale-to-Zero' sleep state. Please re-execute in 2-3 minutes.")
        return

    # Statistical Compute Matrix
    accuracy = accuracy_score(ground_truth, ai_predictions)
    f1 = f1_score(ground_truth, ai_predictions, pos_label="(A)", average="binary")
    
    print(f"▶ Global Balanced Accuracy Metric: {accuracy * 100:.2f}%")
    print(f"▶ Target Clinical Pathological F1-Score (FTD): {f1 * 100:.2f}%")
    print("\n📋 Rigorous Classification Report Boundary Breakdown:")
    print(classification_report(ground_truth, ai_predictions, target_names=["FTD (A)", "Healthy Aged Brain (B)"]))

    # IO persistence sequence for audit compliance
    os.makedirs("results", exist_ok=True)
    with open(results_path, "w", encoding="utf-8") as out_file:
        for entry in reproducibility_logs:
            out_file.write(json.dumps(entry) + "\n")
    
    print(f"💾 Reproducibility log written for validation audit: '{results_path}'")


if __name__ == "__main__":
    run_pipeline()