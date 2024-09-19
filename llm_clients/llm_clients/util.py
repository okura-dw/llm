import os

import demucs.separate


def vocal_extract(audio_path: str, dirname: str) -> str:
    """ボーカル抽出してファイルのパスを返す

    audio_path
        ボーカル抽出したい音声ファイルのパス
    dirname
        ボーカルのファイルを置きたいパス {dirname}/htdemucs/ 以下に作られる
    """
    audio_fname, _ = os.path.splitext(os.path.basename(audio_path))
    vocal_path = f"{dirname}/htdemucs/{audio_fname}/vocals.wav"
    if not os.path.exists(vocal_path):
        demucs.separate.main(["--two-stems", "vocals", audio_path, "-o", dirname])
    return vocal_path
