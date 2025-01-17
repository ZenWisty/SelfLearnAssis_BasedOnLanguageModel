
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
    file_path = r"E:\BiliBiliVideo\社会\探访东德高级人才住宅区，美国黑人区顶流up吃上德国甜甜圈！feat.街头小小小霸王\14-探访东德高级人才住宅区，美国黑人区顶流up吃上德国甜甜圈！feat.街头小小小霸王-480P 清晰-AVC.mp3"
    output_path = r'E:\BiliBiliVideo\社会\探访东德高级人才住宅区，美国黑人区顶流up吃上德国甜甜圈！feat.街头小小小霸王\14-探访东德高级人才住宅区，美国黑人区顶流up吃上德国甜甜圈！feat.街头小小小霸王-480P 清晰-AVC.txt'

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