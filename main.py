import os
import json
import random
import torch
import evaluate
from tqdm import tqdm
from dotenv import load_dotenv

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel, get_peft_model_state_dict

from utils import *
from optimize import *


load_dotenv()
ACCESS_TOKEN = os.getenv("HF_TOKEN")

# mapping from user-facing names to actual dataset identifiers
DATASET_MAP = {
    "minerva_math": "minerva_math_algebra",
    "xsum":         "EdinburghNLP/xsum"
}


def get_output_name(args):
    """Builds result file path based on method and hyperparameters."""
    base = args.output_dir if args.output_dir else f"result/{args.model_name}/{args.adapter_path}/{args.dataset}"
    if args.method == 'fp':
        return f"{base}/fp_numfewshot{args.num_fewshot}.json"
    elif args.method in ['rtn', 'bin']:
        return f"{base}/{args.method}_{args.num_bits_low}bit_numfewshot{args.num_fewshot}.json"
    elif args.method == 'loraq':
        if args.decomp == 'fft':
            rank_tag = f"ratio{args.ratio}" if args.ratio is not None else f"h{args.rank_high}"
            return f"{base}/loraq_fft_{rank_tag}_{args.num_bits_high}_opt{args.opt}_Bcol{args.along_column_B}_Acol{args.along_column_A}_numfewshot{args.num_fewshot}.json"
        elif args.decomp == 'svd':
            rank_tag = f"ratio{args.ratio}" if args.ratio is not None else f"h{args.rank_high}"
            return f"{base}/loraq_svd_{rank_tag}_{args.num_bits_high}_opt{args.opt}_Bcol{args.along_column_B}_Acol{args.along_column_A}_numfewshot{args.num_fewshot}.json"
        else:
            return f"{base}/loraq_{args.decomp}_h{args.rank_high}_{args.num_bits_high}_opt{args.opt}_numfewshot{args.num_fewshot}.json"


def parse_args():
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", required=True,
                        choices=["meta-llama/Llama-2-7b-hf", "mistralai/Mistral-7B-v0.1", "meta-llama/Llama-2-13b-hf"])
    parser.add_argument("--adapter_path", required=True, type=str)
    parser.add_argument("--dataset", type=str, default='gsm8k',
                        choices=["gsm8k", "minerva_math", "xsum"])
    parser.add_argument("--method", required=True,
                        choices=['fp', 'rtn', 'bin', 'loraq'])
    parser.add_argument("--decomp", type=str, default=None,
                        choices=['svd', 'fft', 'random', 'norm'])
    parser.add_argument("--split", type=str, default='sqrt', choices=['sqrt', 'B', 'A'])
    parser.add_argument("--num_bits_high", type=int)
    parser.add_argument("--num_bits_low", type=int)
    parser.add_argument("--group_size", type=int, default=128)
    parser.add_argument("--rank_high", type=int, default=4)
    parser.add_argument("--ratio", type=float, default=None)
    parser.add_argument("--num_fewshot", type=int, default=0)
    parser.add_argument("--along_column_B", action='store_true')
    parser.add_argument("--along_column_A", action='store_true')
    parser.add_argument("--opt", action='store_true')
    parser.add_argument("--output_dir", type=str, default=None)

    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(42)

    # resolve user-facing dataset name to actual identifier
    actual_dataset = DATASET_MAP.get(args.dataset, args.dataset)

    # --- load base model in 4-bit NF4, stays frozen throughout ---
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, quantization_config=bnb_config, token=ACCESS_TOKEN
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, token=ACCESS_TOKEN)

    # load LoRA adapter on top of frozen base model
    model = PeftModel.from_pretrained(model, args.adapter_path, local_files_only=True)
    model.eval()

    # apply quantization to LoRA weights 
    lora_state_dict = get_peft_model_state_dict(model)

    if args.method == 'fp':
        # full precision baseline - no quantization
        pass

    elif args.method in ['rtn', 'bin']:
        # uniform quantization applied to all LoRA A and B matrices
        for key in lora_state_dict:
            if 'lora_B' in key:
                continue

            lora_A_name = key.replace('lora_A.weight', 'lora_A.default')
            lora_B_name = key.replace('lora_A.weight', 'lora_B.default')

            lora_B = model.get_submodule(lora_B_name).weight
            lora_A = model.get_submodule(lora_A_name).weight

            lora_B.data = quant(lora_B, group_size=args.group_size, num_bits=args.num_bits_low, method=args.method)
            lora_A.data = quant(lora_A, group_size=args.group_size, num_bits=args.num_bits_low, method=args.method)

    elif args.method == 'loraq':
        if args.decomp is None:
            raise ValueError("--decomp is required when --method is loraq")

        total_bits, total_params = 0, 0

        for key in tqdm(lora_state_dict):
            if 'lora_B' in key:
                continue

            lora_A_name = key.replace('lora_A.weight', 'lora_A.default')
            lora_B_name = key.replace('lora_A.weight', 'lora_B.default')

            lora_B = model.get_submodule(lora_B_name).weight
            lora_A = model.get_submodule(lora_A_name).weight

            if args.decomp == 'fft':
                B, A, num_bits, num_params = lora_quant_fft(
                    lora_B, lora_A, bits_hi=args.num_bits_high, gs=args.group_size,
                    col_B=args.along_column_B, col_A=args.along_column_A,
                    ratio=args.ratio, rh=args.rank_high
                )
            else:
                B, A, num_bits, num_params = lora_quant_svd(
                    lora_B, lora_A, decomp=args.decomp, rh=args.rank_high,
                    bits_hi=args.num_bits_high, gs=args.group_size,
                    col_B=args.along_column_B, col_A=args.along_column_A,
                    split=args.split, ratio=args.ratio, opt=args.opt
                )

            total_bits   += num_bits
            total_params += num_params

            lora_B.data = B.clone()
            lora_A.data = A.clone()

        print(f"avg bits = {total_bits / total_params:.4f} | decomp = {args.decomp} | ratio = {args.ratio} | model = {args.model_name}")

    # --- evaluation ---
    output_path = get_output_name(args)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if actual_dataset == 'EdinburghNLP/xsum':
        # summarization - evaluate with ROUGE-L
        ds = load_dataset("EdinburghNLP/xsum")['test']
        gens, refs = [], []

        for data in tqdm(ds):
            prompt  = f"Document : {data['document']} \n\n Summary : "
            inputs  = tokenizer(prompt, return_tensors="pt").to(model.device)
            outputs = model.generate(**inputs, max_new_tokens=128, do_sample=False)

            # strip prompt tokens, decode only newly generated tokens
            new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
            refs.append(data['summary'])
            gens.append(tokenizer.decode(new_tokens, skip_special_tokens=True))

        results = evaluate.load("rouge").compute(predictions=gens, references=refs)
        print(results)

        with open(output_path, "w") as f:
            json.dump({'results': results, 'gens': gens, 'refs': refs}, f, indent=2)

    else:
        # math reasoning - uses lm-eval harness (gsm8k, minerva_math)
        from lm_eval import evaluator
        from lm_eval.models.huggingface import HFLM

        hf_lm   = HFLM(pretrained=model, tokenizer=tokenizer, device="cuda")
        results = evaluator.simple_evaluate(
            model=hf_lm,
            tasks=[actual_dataset],
            num_fewshot=args.num_fewshot,
            batch_size=1,
            device="cuda"
        )

        with open(output_path, "w") as f:
            json.dump({'results': results['results'], 'gens': results['samples'][actual_dataset]}, f, indent=2)


if __name__ == "__main__":
    main()


# python main.py --model_name meta-llama/Llama-2-7b-hf --adapter_path ./weights/llama7b_metamath_lora16/checkpoint-24688 --dataset gsm8k --method fp --num_fewshot 0
# python main.py --model_name meta-llama/Llama-2-7b-hf --adapter_path ./weights/llama7b_metamath_lora16/checkpoint-24688 --dataset gsm8k --method loraq --decomp svd --ratio 0.9 --num_bits_high 3 --along_column_B --opt
# python main.py --model_name meta-llama/Llama-2-7b-hf --adapter_path ./weights/llama7b_metamath_lora16/checkpoint-24688 --dataset gsm8k --method loraq --decomp fft --ratio 0.9 --num_bits_high 3 --along_column_B
