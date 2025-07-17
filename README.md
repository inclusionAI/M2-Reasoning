# M2-Reasoning: Empowering MLLMs with Unified General and Spatial Reasoning

📖 [Technical Report](./assets/M2-Reasoning.pdf) | 📄 [arXiv](https://arxiv.org/abs/2507.08306) | 🤗 [Hugging Face](https://huggingface.co/inclusionAI/M2-Reasoning)｜ 🤖 [ModelScope](https://www.modelscope.cn/models/inclusionAI/M2-Reasoning)

## Introduction

We introduce M2-Reasoning-7B, a model designed to excel in both general and spatial reasoning. Our approach integrates two key innovations: (1) a novel data pipeline that generates 294.2K high-quality data samples (168K for cold-start fine-tuning and 126.2K for RLVR), which feature logically coherent reasoning trajectories and have undergone comprehensive assessment; and (2) a dynamic multi-task training strategy with step-wise optimization to mitigate conflicts between data, and task-specific rewards for delivering tailored incentive signals. This combination of curated data and advanced training allows M2-Reasoning-7B to set a new state-of-the-art (SOTA) across 8 benchmarks, showcasing superior performance in both general and spatial reasoning domains.
![](assets/teaser.png)

## 📌 Updates

- [2025.07.14] 🔥 Our Technical Report is available on 📄 [arXiv](https://arxiv.org/abs/2507.08306).
- [2025.07.11] 🔥 We release M2-Reasoning on 🤗 [Hugging Face](https://huggingface.co/inclusionAI/M2-Reasoning) and 🤖 [ModelScope](https://www.modelscope.cn/models/inclusionAI/M2-Reasoning).

## Key Features

- A High-quality Data Construction Pipeline: We design and implement a multi-stage data synthesis and curation pipeline that generates vast amounts of reasoning data.
- A Dynamic Multi-Task Training Strategy: We propose a sophisticated training strategy that effectively handles data heterogeneity. It features step-wise dynamic optimization to mitigate conflicts between different data sources and a task-specific reward formulation to provide tailored incentive signals.
- Unified General and Spatial Reasoning Model: We propose M2-Reasoning-7B, an MLLM uniquely engineered for both abstract and spatial reasoning. Extensive evaluations on 8 distinctbenchmarks demonstrate that, by leveraging our custom data and training pipelines, M2-Reasoning establishes new state-of-the-art (SOTA) results across both general and spatial reasoning domains.

## Evaluation

We conduct a comprehensive evaluation of our models across two key domains: general and spatial
reasoning. Our evaluation utilizes a diverse set of public benchmarks, grouped by the primary
capability they measure:

- General Reasoning (Mathematical & Logical): To evaluate this capability, we employ six benchmarks: MathVista, MathVision, MathVerse, DynaMath, WeMath, and LogicVista.

|Models| MathVista| MathVision| MathVerse| DynaMath| WeMath| LogicVista| Avg. (Δ)|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|***Base-Scale General Models***|
|InternVL3-8B | 70.5| 30.0| 38.5| 25.7 |39.5 |44.5 |41.4|
|InternVL3-9B | 69.0 | 29.3| 37.9 |25.1 |34.8| 49.0 |40.8|
|Qwen2.5-VL-7B |68.1 |25.4 |41.1 |21.8 |36.2| 47.9| 40.1|
|MUG-U-7B | 74.8 |26.1 |35.4 |17.2 |26.5 |39.8| 36.6|
|SAIL-VL-1.6-8B | 74.2 |23.2| 33.4 |14.0 |29.6 |41.4| 36.0|
|***Base-Scale Reasoning Models***|
|WeThink-VL-7B| 71.6 |26.0| 44.2 |24.8 |**48.0** |**51.2**| 44.3 (+4.2)|
|Taichu-VLR-7B | 72.3| 27.1 |46.7 |23.0 |44.0 |48.3 |43.6|
|VLAA-Thinker-7B | 68.0 |26.4| **48.2** |22.4 |41.5 |48.5 |42.5 (+2.4)|
|URSA-8B-PS-GRPO | 67.8 |**31.8** |41.5 |22.4| 38.3 |44.7 |41.1 (+8.2)|
|Ovis2-8B |71.8 |25.9| 42.3 |20.4 |27.2 |39.4| 37.8|
|***Our Models***|
|Base Model |70.2| 25.9| 30.5| 20.2| 27.2| 37.8| 35.5|
|M2-Reasoning-CI-7B|  71.7| 29.2| 42.1| 25.0 |42.8| 46.8 |42.9 (+7.4)|
|M2-Reasoning-7B | **75.0** |31.5| 44.7 |**26.8** |41.8 |50.0 |**45.0 (+9.5)**|
|M2-Reasoning-7B-HF* | 74.7 |30.5| 46.1 |26.8 |42.7 |49.2 |45.0 (+9.5)|

\* After converting the checkpoints to huggingface, the accuracies are slightly different.

- Spatial Reasoning: We assess this skill using 2 benchmarks: CV-Bench and VSI-Bench
    - CV-Bench:

    | Models | Count | Relation | Depth | Distance | Avg. |
    | :--- | :---: | :---: | :---: | :---: | :---: |
    | ***Large-Scale Models*** | | | | | |
    | GPT-4O | 65.9 | 85.7 | 87.8 | 78.2 | 78.9 |
    | Gemini-1.5-pro | 70.4 | 85.2 | 82.4 | 72.8 | 77.4 |
    | ***Base-Scale Models*** | | | | | |
    | InternVL3-8B| **74.0** |  90.6  |  84.3  |  81.0  |  82.0  |
    | Qwen2.5-VL-7B-Instruct | 65.2 |  86.6  | 70.6 | 79.8 | 75.0 |
    | LLava-NEXT-Video-7B  | 59.3 | 77.0 | 71.3 | 54.7 | 65.2 |
    | ***Our Models*** | | | | | |
    | M2-Reasoning-7B |  66.6  | **92.8** | **89.3** | **84.3** | **82.3** |

    - VSI-Bench:

    | | OC | AD| OS|RS |RDs |RDr |RP |AO |Avg. |
    | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
    | ***Large-Scale Models*** | | | | | | | | | |
    | Gemini-1.5-pro  | 56.2 | 30.9 | 64.1 | 43.6 | 51.3 | 46.3 | 36.0 | 34.6 | 45.4 |
    | GPT-4O  | 46.2 | 5.3 | 43.8 | 38.2 | 37.0 | 41.3 | 31.5 | 28.5 | 34.0 |
    | ***Base-Scale Models*** | | | | | | | | | |
    | InternVL3-8B | **68.1** | **39.0** | 48.4 | 33.6 | **48.3** | 36.4 | 27.3 | **35.4** |  42.1  |
    | Video-R1-7B | - | - | - | - | - | - | - | - | 37.1 |
    | Qwen2.5-VL-7B-Instruct| 37.7 | 20.1 | 49.7  |  37.4  | 38.5 | 40.4 |  31.4  |  32.0  | 35.9 |
    | LLava-NeXT-Video-7B|  48.5  | 14.0 |  47.8  | 24.2 |  43.5  | 42.4 |  **34.0**  | 30.6 | 35.6 |
    | ***Our Models*** | | | | | | | | | |
    | M2-Reasoning-7B | 41.0 |  34.0  | **60.9** | **55.4** | 40.7 | **47.3** | 29.9 | 28.8 | **42.3** |

## Model Downloads
You can download the model from both 🤗 [Hugging Face](https://huggingface.co/inclusionAI/M2-Reasoning) and 🤖 [ModelScope](https://www.modelscope.cn/models/inclusionAI/M2-Reasoning).

## Installation

Please download our model following Model Downloads, then you can refer to the following codes to run M2-Reasoning model.
The basic environment is `python=3.10`, `torch=2.6.0+cu124`, `transformers=4.49.0`
## Example Usage

We provide a small example on the usage of this repo. For detailed usage.

``` python
import os
import torch

from transformers import (
    AutoProcessor,
    AutoTokenizer,
)

import warnings
import argparse
from modeling_bailing_qwen2_5 import Bailing_qwen2_5NativeForConditionalGeneration
from processing_bailing_qwen2_5 import Bailing_qwen2_5Processor

warnings.filterwarnings("ignore")

class BailingMMInfer:
    def __init__(self,
        model_name_or_path,
        device="cuda",
        max_pixels=None,
        min_pixels=None,
        video_max_pixels=768 * 28 * 28,
        video_min_pixels=128 * 28 * 28,
        generation_config=None
    ):
        super().__init__()
        self.model_name_or_path = model_name_or_path

        self.device = device

        self.device_map = device

        self.video_max_pixels = video_max_pixels if video_max_pixels is not None else 768 * 28 * 28
        self.video_min_pixels = video_min_pixels if video_min_pixels is not None else 128 * 28 * 28

        self.model, self.tokenizer, self.processor = self.load_model_processor()
        if max_pixels is not None:
            self.processor.max_pixels = max_pixels
        if min_pixels is not None:
            self.processor.min_pixels = min_pixels
        if generation_config is None:
            generation_config = {
                "num_beams": 1,
                "do_sample": True,
                "temperature": 0.9
            }

        self.generation_config = generation_config


    def load_model_processor(self):
        
        model = Bailing_qwen2_5NativeForConditionalGeneration.from_pretrained(
            self.model_name_or_path,
            torch_dtype=torch.bfloat16,
            device_map=self.device_map,
            _attn_implementation="flash_attention_2"
        ).eval()

        tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, add_bos_token=True, trust_remote_code=True)
        processor = Bailing_qwen2_5Processor.from_pretrained(self.model_name_or_path, trust_remote_code=True)

        return model, tokenizer, processor

    def generate(self, messages, max_new_tokens=512):
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, use_system=True
        )

        image_inputs, video_inputs = self.processor.process_vision_info(messages)


        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            return_tensors="pt",
        )
        # print(inputs)
        print(self.tokenizer.decode(inputs['input_ids'][0]))

        inputs = inputs.to(self.device)

        for k in inputs.keys():
            if k == "pixel_values" or k == "pixel_values_videos":
                inputs[k] = inputs[k].to(dtype=torch.bfloat16)

        with torch.no_grad():
            generated_ids = self.model.generate(
                inputs,
                max_new_tokens=max_new_tokens,
                eos_token_id=self.processor.tokenizer.eos_token_id,
                **self.generation_config,
            )

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=False, clean_up_tokenization_spaces=False
        )[0]

        return output_text

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', type=str, default="inclusionAI/M2-Reasoning")
    parser.add_argument('--max_pixels', type=int, default=401408)
    parser.add_argument('--min_pixels', type=int, default=401408)
    parser.add_argument('--max_new_tokens', type=int, default=4096)

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # model_name_or_path = os.path.join(args.input_dir, args.model_name_or_path)
    bailing2 = BailingMMInfer(
        args.model_name_or_path, 
        device=device, 
        max_pixels=args.max_pixels, 
        min_pixels=args.min_pixels
    )

    messages = [
        {
            "role": "system", 
            "content": [
                {"type": "text", "text": "You are a helpful assistant. When the user asks a question, your response must include two parts: first, the reasoning process enclosed in <think>...</think> tags, then the final answer enclosed in <answer>...</answer> tags. The critical answer or key result should be placed within \\boxed{}."}]},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": "./assets/example1.png"},
                {"type": "text", "text": "\nQuestion:\n\nRhombus $QRST$ has an area of 137.9 square meters. If $RT$ is 12.2 meters, find $QS$.\nA. 11.3\nB. 22.4\nC. 22.6\nD. 25.6"},
            ],
        },
    ]
    output_text = bailing2.generate(messages, max_new_tokens=args.max_new_tokens)
    print(output_text)



'''
[Output]:

<think>
To find the length of \( QS \) in the rhombus \( QRST \), we can use the formula for the area of a rhombus, which is given by:

\[
\text{Area} = \frac{1}{2} \times d_1 \times d_2
\]

where \( d_1 \) and \( d_2 \) are the lengths of the diagonals. In this problem, we are given:
- The area of the rhombus is 137.9 square meters.
- One of the diagonals, \( RT \), is 12.2 meters.

We need to find the length of the other diagonal, \( QS \).

Let's denote:
- \( d_1 = RT = 12.2 \) meters
- \( d_2 = QS \)

Substitute the known values into the area formula:

\[
137.9 = \frac{1}{2} \times 12.2 \times QS
\]

To solve for \( QS \), first multiply both sides by 2 to eliminate the fraction:

\[
275.8 = 12.2 \times QS
\]

Next, divide both sides by 12.2:

\[
QS = \frac{275.8}{12.2}
\]

Now, perform the division:

\[
QS \approx 22.6
\]

So, the length of \( QS \) is approximately 22.6 meters.

Looking at the options provided:
A. 11.3
B. 22.4
C. 22.6
D. 25.6

The correct answer is C. 22.6.
</think>
<answer>
\boxed{C. 22.6}
</answer><|im_end|>
'''
```

## License and Legal Disclaimer

This code repository is licensed under the MIT License, and the Legal Disclaimer is located in the LEGAL.md file under the project's root directory.

## Citation

If you find our work helpful, feel free to give us a cite.

```
@misc{M2reasoning2025,
      title  = {M2-Reasoning: Empowering MLLMs with Unified General and Spatial Reasoning}, 
      author = {Inclusion AI},
      year = {2025},
      archivePrefix = {arXiv},
}
```
