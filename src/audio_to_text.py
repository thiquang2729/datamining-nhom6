# ==========================================
# audio_to_text.py
# PhoWhisper Speech To Text
# ==========================================

import os
from transformers import pipeline


class PhoWhisperSTT:

    def __init__(self,
                 model_name="vinai/PhoWhisper-base"):

        print("Đang tải mô hình PhoWhisper...")
        self.asr = pipeline(
            task="automatic-speech-recognition",
            model=model_name
        )

        print("Tải mô hình thành công!")

    def transcribe(self, audio_path):

        if not os.path.exists(audio_path):
            raise FileNotFoundError(
                f"Không tìm thấy file: {audio_path}"
            )

        print(f"\nĐang xử lý: {audio_path}")

        result = self.asr(audio_path)

        return result["text"]

    def save_text(self,
                  text,
                  output_path="data/transcript.txt"):

        os.makedirs(
            os.path.dirname(output_path),
            exist_ok=True
        )

        with open(
            output_path,
            "w",
            encoding="utf-8"
        ) as f:
            f.write(text)

        print(
            f"Đã lưu kết quả vào {output_path}"
        )


# ==========================================
# MAIN
# ==========================================

if __name__ == "__main__":

    AUDIO_FILE = "data/sample_audio.wav"

    stt = PhoWhisperSTT()

    text = stt.transcribe(AUDIO_FILE)

    print("\n====================")
    print("KẾT QUẢ NHẬN DẠNG")
    print("====================")
    print(text)

    stt.save_text(
        text,
        "data/transcript.txt"
    )