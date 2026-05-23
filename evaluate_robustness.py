"""
Longevity-LLM Robustness & Stress-Testing Framework
Ablation Study: Evaluating Diagnostic Accuracy Under Biomarker Omission
"""

import json
import re
import os
import requests
from sklearn.metrics import accuracy_score, f1_score, classification_report
from concurrent.futures import ThreadPoolExecutor, as_completed

ENDPOINT_URL = "https://swchnq0ekc3scmqw.us-east-2.aws.endpoints.huggingface.cloud/v1/chat/completions"
ACCESS_TOKEN = "hf_qAwEwEmwtVTbPzSWKVuCoeXrfdTsROvgt"

ground_truth = []
ai_predictions = []
robustness_logs = []

def extract_answer_letter(text: str) -> str:
    if not text:
        return None
    matches = re.findall(r"(?<![A-Za-z])([A-F])(?![A-Za-z])", text)
    if matches:
        return f"({matches[-1].upper()})"
    return None

def apply_text_ablation(messages):
    """
    Simulates real-world data corruption by omitting definitive clinical biomarker keywords.
    Forces the LLM to rely on subtle secondary multi-omic patterns.
    """
    corrupted_messages = []
    # Target biomarkers to hide from the model for the stress test
    blacklisted_words = ["TDP-43", "TDP43", "C9orf72", "Granulin", "GRN", "Pick bodies"]
    
    for msg in messages:
        content = msg["content"]
        if msg["role"] == "user":
            # Replace case-insensitively with a generic mask
            for word in blacklisted_words:
                content = re.sub(word, "[OMITTED_BIOMARKER]", content, flags=re.IGNORECASE)
        
        corrupted_messages.append({"role": msg["role"], "content": content})
    
    return corrupted_messages

def evaluate_sample_robustness(json_line: str):
    raw_data = json.loads(json_line)
    task_id = raw_data.get("lb_id", "UNKNOWN")
    messages = raw_data["messages"]
    
    gold_answer = messages[-1]["content"].strip().upper()
    
    # APPLY STRESS TEST: Injecting noise/missing data
    prompt_messages = apply_text_ablation(messages[:-1])
    
    headers = {
        "Authorization": f"Bearer {ACCESS_TOKEN}",
        "Content-Type": "application/json",
        "Accept": "application/json"
    }
    
    payload = {
        "model": "longevity-llm",
        "messages": prompt_messages,
        "max_tokens": 50,
        "temperature": 0.01,
        "chat_template_kwargs": {"enable_thinking": False}
    }
    
    try:
        response = requests.post(ENDPOINT_URL, headers=headers, json=payload, timeout=60)
        
        if response.status_code != 200:
            return gold_answer, None, None
            
        response_json = response.json()
        raw_content = response_json["choices"][0]["message"]["content"]
        
        predicted_letter = extract_answer_letter(raw_content.strip())
        
        log_entry = {
            "lb_id": task_id,
            "gold": gold_answer,
            "predicted": predicted_letter,
            "raw_response": raw_content.strip()
        }
        
        return gold_answer, predicted_letter, log_entry
        
    except Exception:
        return gold_answer, None, None

def run_robustness_pipeline():
    dataset_path = "data/dft_aged_brain_benchmark.jsonl"
    results_path = "results/robustness_stress_test_log.jsonl"
    
    print("🧪 [STRESS TEST] Loading multi-omic clinical matrices...")
    if not os.path.exists(dataset_path):
        print(f"❌ Error: Dataset '{dataset_path}' not found.")
        return

    with open(dataset_path, "r", encoding="utf-8") as file:
        lines = file.readlines()

    print(f"📋 Total test samples loaded: {len(lines)}")
    print("⚠️ Injecting text ablation masks (Hiding primary biomarkers)...")
    print("⚡ Connecting to Hugging Face production cluster...")
    
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(evaluate_sample_robustness, line) for line in lines]
        for future in as_completed(futures):
            gold, pred, log = future.result()
            if log and pred:
                ground_truth.append(gold)
                ai_predictions.append(pred)
                robustness_logs.append(log)

    print("\n========================================================")
    print("   🛡️ ROBUSTNESS & ABLATION STRESS REPORT - RESULTS")
    print("========================================================")
    
    if not ground_truth or not ai_predictions:
        print("❌ Error: Server instance cold-start or connection failure. Retry in 2 minutes.")
        return

    accuracy = accuracy_score(ground_truth, ai_predictions)
    f1 = f1_score(ground_truth, ai_predictions, pos_label="(A)", average="binary")
    
    print(f"▶ Degraded Grid Accuracy: {accuracy * 100:.2f}%")
    print(f"▶ Degraded F1-Score (FTD Target): {f1 * 100:.2f}%")
    print("\n📋 Detailed Stress-Testing Matrix Breakdown:")
    print(classification_report(ground_truth, ai_predictions, target_names=["FTD (A)", "Healthy Aged Brain (B)"]))

    os.makedirs("results", exist_ok=True)
    with open(results_path, "w", encoding="utf-8") as out_file:
        for entry in robustness_logs:
            out_file.write(json.dumps(entry) + "\n")
    
    print(f"💾 Robustness logs successfully saved to: '{results_path}'")

if __name__ == "__main__":
    run_robustness_pipeline()