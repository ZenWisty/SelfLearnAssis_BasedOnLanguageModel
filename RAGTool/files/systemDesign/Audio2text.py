
import librosa
from transformers import pipeline


def load_and_preprocess_mp3(file_path):
    # 读取 MP3 文件
    audio_data, sample_rate = librosa.load(file_path, sr=None)  # sr=None 保持原始采样率
    # 这里假设 pipeline 期望的采样率是 16000，可根据实际情况调整
    target_sample_rate = 16000
    if sample_rate!= target_sample_rate:
        # 重采样到目标采样率
        audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=target_sample_rate)
    return audio_data


def main():
    file_path = r"E:\Python_work\LLM_MetaGPT\Publish\SelfLearnAssis_BasedOnLanguageModel\RAGTool\files\systemDesign\audioData\8个常见的系统设计概念\1-8个常见的系统设计概念-480P 清晰-AVC.mp3"
    output_path = r'E:\Python_work\LLM_MetaGPT\Publish\SelfLearnAssis_BasedOnLanguageModel\RAGTool\files\systemDesign\TextData\8个常见的系统设计概念.txt'

    asr_pipeline = pipeline(task="automatic-speech-recognition",
               model = "distil-whisper/distil-small.en")

    audio_data = load_and_preprocess_mp3(file_path)

    # check sampling rate
    # asr_pipeline.feature_extractor.sampling_rate
    result = asr_pipeline(audio_data, return_timestamps=True)
    with open(output_path, 'w') as f:
        f.write(result['text'])


if __name__ == '__main__':
    main()