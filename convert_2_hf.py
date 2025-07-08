import argparse
import torch
import warnings
# from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from transformers import (
    AutoProcessor, 
    AutoConfig,
    AutoModel,
    AutoTokenizer,
)
# from transformers.models.bailingmm_qwen2_5.processing_bailing_qwen2_5 import Bailing_qwen2_5Processor
# from transformers.models.bailingmm_qwen2_5.modeling_qwen2_5 import Qwen2ForCausalLM
# from transformers.models.bailingmm_qwen2_5.qwen2_5_vit import Qwen2_5_VisionTransformer
from modeling_bailing_qwen2_5 import  Bailing_qwen2_5NativeForConditionalGeneration
from configuration_bailing_qwen2_5 import Bailing_qwen2_5Config

from transformers.models.bailingmm_qwen2_5.processing_bailing_qwen2_5 import (
    DEFAULT_IMAGE_PATCH_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    DEFAULT_VID_START_TOKEN,
    DEFAULT_VID_END_TOKEN,
    DEFAULT_GEN_IMAGE_PATCH_TOKEN,
    DEFAULT_GEN_IM_START_TOKEN,
    DEFAULT_GEN_IM_END_TOKEN,
    PLACEHOLDER_IMAGE_TOKEN_IN_TEXT,
    DEFAULT_END_OF_CHUNK_TOKEN,
    DEFAULT_END_OF_AUDIO_TOKEN,
    DEFAULT_AUDIO_PATCH_TOKEN,
    DEFAULT_AU_START_TOKEN,
    DEFAULT_AU_END_TOKEN,
    DEFAULT_GEN_AUDIO_PATCH_TOKEN,
    DEFAULT_GEN_AU_START_TOKEN,
    DEFAULT_GEN_AU_END_TOKEN,
    PLACEHOLDER_AUDIO_TOKEN_IN_TEXT,
    DEFAULT_ASR_TOKEN,
    DEFAULT_TTS_TOKEN,
)
from collections import OrderedDict


BAILING_MM_TOKENS = [
    DEFAULT_IMAGE_PATCH_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    DEFAULT_VID_START_TOKEN,
    DEFAULT_VID_END_TOKEN,
    DEFAULT_GEN_IMAGE_PATCH_TOKEN,
    DEFAULT_GEN_IM_START_TOKEN,
    DEFAULT_GEN_IM_END_TOKEN,
    PLACEHOLDER_IMAGE_TOKEN_IN_TEXT,
    DEFAULT_END_OF_CHUNK_TOKEN,
    DEFAULT_END_OF_AUDIO_TOKEN,
    DEFAULT_AUDIO_PATCH_TOKEN,
    DEFAULT_AU_START_TOKEN,
    DEFAULT_AU_END_TOKEN,
    DEFAULT_GEN_AUDIO_PATCH_TOKEN,
    DEFAULT_GEN_AU_START_TOKEN,
    DEFAULT_GEN_AU_END_TOKEN,
    PLACEHOLDER_AUDIO_TOKEN_IN_TEXT,
    DEFAULT_ASR_TOKEN,
    DEFAULT_TTS_TOKEN,
]

KEYS_TO_MODIFY_MAPPING = {
    # vision_model
    "eva_encoder.": "vision_model.",
    # audio_model
    "audio_encoder.": "audio_model.encoder.",
    # language_model
    "glm_model.": "language_model.",
    # experts mlp
    "experts.w": "experts.mlp.w"
}

def convert_state_dict_to_hf(state_dict, hf_state_dict):
    new_state_dict = {}
    missing_keys = []
    unexpected_keys = []
    
    new_state_dict = OrderedDict()
    type_dict = {}

    for key, value in state_dict.items():
        hf_key = key
        for key_to_modify, new_key in KEYS_TO_MODIFY_MAPPING.items():
            hf_key = hf_key.replace(key_to_modify, new_key)
        
        if "audio" in hf_key:
            continue
        else:
            new_state_dict[hf_key] = value
            type_dict[str(value.dtype)] = 1
    
    # for key, value in state_dict.items():        
    #     for key_to_modify, new_key in KEYS_TO_MODIFY_MAPPING.items():
    #         if key_to_modify in key:
    #             key = key.replace(key_to_modify, new_key)

    #     if key not in hf_state_dict.keys() or "audio" in key:
    #         unexpected_keys.append(key)
    #         state_dict.pop(key)
    #         continue
    #     new_state_dict[key] = value
    # for key in hf_state_dict.keys():
    #     if key not in new_state_dict.keys():
    #         missing_keys.append(key)
    print(f"missing_keys: {missing_keys}")
    print(f"unexpected keys: {unexpected_keys}")
    print(type_dict)
    return new_state_dict

warnings.filterwarnings('ignore')
# /mnt/bailingmm-chat/ckpt/yuanqing.mzp/checkpoint/save/20250630_reasoning_rl/20250628_math5w_ratio2_spatial_0627_60k_from_0627_baseline394_addtrain-mpl1024-mcl1024-n16-1ep_unfrzVit_lr1e-6_kl1e-2_gpu128/checkpoint-1713/pytorch_model.bin
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_path', type=str, default="/mnt/bailingmm-chat/ckpt/yuanqing.mzp/checkpoint/save/20250630_reasoning_rl/20250628_math5w_ratio2_spatial_0627_60k_from_0627_baseline394_addtrain-mpl1024-mcl1024-n16-1ep_unfrzVit_lr1e-6_kl1e-2_gpu128/checkpoint-1713/")
    parser.add_argument('--dest_dir', type=str, default="/mllm_native/rulixiang.rlx/Ming-Reasoning/Ming-Reasoning/")
    parser.add_argument('--llm_path', type=str, default="/mllm_native/rulixiang.rlx/ckpts_and_pkgs/Qwen2.5-7B-Instruct")
    parser.add_argument('--vision_path', type=str, default="/mllm_native/rulixiang.rlx/ckpts_and_pkgs//Qwen2-5-ViT-600m")
    args = parser.parse_args()
    # audio_config = GLMAudioConfig(checkpoint_activations=True)
    llm_config = AutoConfig.from_pretrained("/video_hy2/workspace/rulixiang.rlx/transformers-4.49.0/saved_glm_model/", _attn_implementation="flash_attention_2", trust_remote_code=True)
    vision_config = AutoConfig.from_pretrained("/video_hy2/workspace/rulixiang.rlx/transformers-4.49.0/saved_vision_model/", _attn_implementation="flash_attention_2", trust_remote_code=True)

    config = Bailing_qwen2_5Config(
        mlp_depths=2,
        llm_config=llm_config.to_dict(),
        vision_config=vision_config.to_dict(),
    )

    model = Bailing_qwen2_5NativeForConditionalGeneration(config).to(torch.bfloat16).eval()
    state_dict = torch.load(args.ckpt_path + "pytorch_model.bin")#['model']

    new_state_dict = convert_state_dict_to_hf(state_dict, model.state_dict())

    msg = model.load_state_dict(new_state_dict, strict=False)
    print(msg)
    ref_state_dict = model.state_dict()

    state_dict, ref_state_dict = ref_state_dict, state_dict

    # for key in state_dict.keys():
    #     new_key = key
    #     for kk in KEYS_TO_MODIFY_MAPPING.keys():
    #         new_key = new_key.replace(KEYS_TO_MODIFY_MAPPING[kk], kk)
    #     # print(new_key)

    #     if not torch.equal(ref_state_dict[new_key].cpu(), state_dict[key].cpu().to(ref_state_dict[new_key].dtype)):
    #         print(new_key)

    print(args.dest_dir)
    model.save_pretrained(args.dest_dir, safe_serialization=True)

