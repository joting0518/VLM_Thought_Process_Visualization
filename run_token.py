import torch
from transformers import LogitsProcessorList, TopKLogitsWarper
import argparse
import re
from io import BytesIO
import os, os.path as osp
import requests
import torch
from PIL import Image
import torch.nn.functional as F
from llava.constants import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                             DEFAULT_IMAGE_TOKEN, IMAGE_PLACEHOLDER,
                             IMAGE_TOKEN_INDEX)
from llava.conversation import SeparatorStyle, conv_templates
from llava.mm_utils import (KeywordsStoppingCriteria, get_model_name_from_path,
                            process_images, tokenizer_image_token)
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
import pandas as pd
import google.generativeai as genai
import os
import csv

def eval_model(args):
    # Model initialization
    disable_torch_init() # 禁用一些PyTorch的操作，漸少GPU的用量
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, model_name, args.model_base)

    csv_file = "/home/nccu/jt/datasets/launch_hit.csv"
    df = pd.read_csv(csv_file)
    outputs_list = []

    for index, row in df.iterrows():
        args.video_file = "/home/nccu/jt/datasets/" + row['video_id']
        args.query = "<video>\n " + row['question'] + " please think step by step."

        print(row['video_id'])
        print(row['question'])

        if args.video_file is None:
            image_files = image_parser(args)
            images = load_images(image_files)
        else:
            if args.video_file.startswith("http") or args.video_file.startswith("https"):
                print("downloading video from url", args.video_file)
                response = requests.get(args.video_file)
                video_file = BytesIO(response.content)
            else:
                assert osp.exists(args.video_file), "video file not found"
                video_file = args.video_file
            from llava.mm_utils import opencv_extract_frames
            images = opencv_extract_frames(video_file, args.num_video_frames) # 從影片中提取固定幀數
      
        qs = args.query
        image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
        if IMAGE_PLACEHOLDER in qs:
            if model.config.mm_use_im_start_end:
                qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
            else:
                qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
        else:
            if DEFAULT_IMAGE_TOKEN not in qs:
                if model.config.mm_use_im_start_end:
                    qs = (image_token_se + "\n") * len(images) + qs
                else:
                    qs = (DEFAULT_IMAGE_TOKEN + "\n") * len(images) + qs

        if "llama-2" in model_name.lower():
            conv_mode = "llava_llama_2"
        elif "v1" in model_name.lower():
            conv_mode = "llava_v1"
        elif "mpt" in model_name.lower():
            conv_mode = "mpt"
        else:
            conv_mode = "llava_v0"

        conv = conv_templates[conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        images_tensor = process_images(images, image_processor, model.config).to(model.device, dtype=torch.float16)
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

        with torch.inference_mode():
            logits_processor = LogitsProcessorList([TopKLogitsWarper(top_k=args.top_k)])
            outputs = model.generate(
                input_ids,
                images=[images_tensor],
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
                use_cache=True,
                stopping_criteria=[stopping_criteria],
                return_dict_in_generate=True,
                output_scores=True,
                logits_processor=logits_processor,
            )

            sequences = outputs.sequences # 最終序列
            scores = outputs.scores  # 每一步生成的 logits

            # print("All candidate sequences:")
            # for i, sequence in enumerate(sequences):
            #     print(f"Sequence {i+1}:")
            #     print(tokenizer.decode(sequence, skip_special_tokens=True))
            token_data = []

            for step, score in enumerate(scores):
                top_k_values, top_k_indices = torch.topk(score, args.top_k, dim=-1) # get top-k token and its probability
                print(f"Step {step + 1}:")
                for i in range(args.top_k):
                    token = tokenizer.decode([top_k_indices[0, i].item()])
                    probability = F.softmax(top_k_values, dim=-1)[0, i].item()
                    print(f"    Token: {token} | Probability: {probability:.4f}")
                    token_data.append({
                        "video_id": row['video_id'],
                        "question": row['question'],
                        "step": step + 1,
                        "token": token,
                        "probability": probability
                    })
        output_csv = "output_tokens.csv"
        keys = token_data[0].keys()
        with open(output_csv, 'a', newline='') as output_file:
            dict_writer = csv.DictWriter(output_file, fieldnames=keys, escapechar='\\', quoting=csv.QUOTE_NONE)
            if output_file.tell() == 0:
                dict_writer.writeheader()  # write header only once
            dict_writer.writerows(token_data) 


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="Efficient-Large-Model/VILA1.5-3b")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-file", type=str, default=None)
    parser.add_argument("--video-file", type=str, default="video")
    parser.add_argument("--num-video-frames", type=int, default=6)
    parser.add_argument("--query", type=str, default="query")
    parser.add_argument("--conv-mode", type=str, default="vicuna_v1")
    parser.add_argument("--sep", type=str, default=",")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--num_beams", type=int, default=5)
    parser.add_argument("--max_new_tokens", type=int, default=50)
    args = parser.parse_args()

    eval_model(args)
