#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
# Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
#  MIT License  (https://opensource.org/licenses/MIT)

from funasr import AutoModel

# model_dir = "iic/SenseVoiceSmall"
model_dir = "iic/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
input_file = "example/zh.mp3"

model = AutoModel(
    model=model_dir,
    trust_remote_code=True,
    vad_model="fsmn-vad",
    punc_model="ct-punc",
    spk_model="cam++",
    disable_pbar=True,
    disable_log=True,
    device="cuda:0",
)

res = model.generate(
    input=input_file, cache={}, language="auto", use_itn=False  # "zn", "en", "yue", "ja", "ko", "nospeech"
)

print(res)
