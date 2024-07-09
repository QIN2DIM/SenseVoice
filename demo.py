#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
# Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
#  MIT License  (https://opensource.org/licenses/MIT)

from model import SenseVoiceSmall


def main():
    model_dir = "iic/SenseVoiceSmall"
    m, kwargs = SenseVoiceSmall.from_pretrained(model=model_dir)

    res = m.inference(
        data_in="example/sent_008-20240617174027-j5sqgxw.wav",
        language="auto",  # "zn", "en", "yue", "ja", "ko", "nospeech"
        use_itn=False,
        **kwargs,
    )

    print(res)


if __name__ == "__main__":
    main()
