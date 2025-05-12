from transformers import AutoModelForCausalLM, AutoTokenizer
from preprocess import preprocess
import torch 
import argparse
import pandas as pd
import json
device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

inference_instruction = '{question} | {answer}' # input format



def inference_prompt(question, upstream_output):
    input = inference_instruction.format(question=question, answer=upstream_output)
    prompt = preprocess(input, "empty")
    return prompt

def extracting(answer_path, task ,target_model="llama2-7b"):
    if task == "truthfulqa":
        df = pd.read_csv(answer_path)
        upstream_outputs = df[target_model]
        questions = df["Question"]

        return questions.to_list(), upstream_outputs.to_list()
    elif task == "harmfulqa":
        with open(answer_path, "r") as file:
            data = json.load(file)
        questions = []
        upstream_outputs = []
        for idx in range(len(data)):
            questions.append(data[idx]["prompt"])
            upstream_outputs.append(data[idx]["response"])
            
        return questions, upstream_outputs


def inference(aligner_path, questions, upstream_outputs, verbose=True):
    # load model
    model = AutoModelForCausalLM.from_pretrained(aligner_path).to(device)
    model.eval()
    tokenizer =AutoTokenizer.from_pretrained(aligner_path, use_fast=False)
    
    ### 데이터 변환
    corrected_outputs = []
    for question, output in zip(questions, upstream_outputs):
        cur_prompt = inference_prompt(question, output)
        split_point = len(cur_prompt)
        
        input_ids = tokenizer(cur_prompt,
                              return_tensors="pt").to(device)
        
        output_ids = model.generate(input_ids=input_ids["input_ids"],
                                    attention_mask=input_ids["attention_mask"],
                                    max_new_tokens=2048,
                                    )[0]
        cur_output = tokenizer.decode(output_ids, skip_special_tokens=True)[split_point:]
        corrected_outputs.append(cur_output)
    
        if verbose:
            print(f'PROMPT: {cur_prompt}')
            print(f'CORRECTION: {cur_output}')

    return corrected_outputs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--aligner_path', type=str, required=False, default="./models/aligner_baseline")
    parser.add_argument('--task', type=str, required=True)
    parser.add_argument('--target_model', type=str, required=True)
    
    args = parser.parse_args()
    if args.aligner_path == "aligner/aligner-7b-v1.0":
        now = "paper"
    else:
        now = "mine" # reproduction
        
    if args.task == "truthfulqa":
        answer_path = "./benchmarks/TruthfulQA/results/answers.csv"
        questions, upstream_outputs = extracting(answer_path, args.task, args.target_model)
        corrected_outputs = inference(args.aligner_path, questions, upstream_outputs)
        df = pd.DataFrame({"Question" : questions,
                            f"corrected_{args.target_model}": corrected_outputs,
                            f'{args.target_model}': upstream_outputs})
        df.to_csv(f'./benchmarks/TruthfulQA/results/corrected_{args.target_model}_{now}.csv', header=True, index=False)
        
    elif args.task == "harmfulqa":
        if args.target_model == "llama2-7b":
            answer_path = "./benchmarks/HarmfulQA/results/harmfulqa_Llama-2-7b-chat-hf_cou_sampled.json"
        elif args.target_model == "vicuna-7b":
            answer_path = "./benchmarks/HarmfulQA/results/harmfulqa_vicuna-7b-v1.5_cou_sampled.json"
        elif args.target_model == "alpaca-7b":
            answer_path = "./benchmarks/HarmfulQA/results/harmfulqa_alpaca-7b-reproduced_cou_sampled.json"
        
        questions, upstream_outputs = extracting(answer_path, args.task, args.target_model)
        corrected_outputs = inference(args.aligner_path, questions, upstream_outputs)
        data = [{"prompt": p, "response": r} for p, r in zip(questions, corrected_outputs)]
        
        with open(f"./benchmarks/HarmfulQA/results/corrected_{args.target_model}_{now}.json", "w", encoding="utf-8") as file:
            json.dump(data, file, indent=2, ensure_ascii=False)
    