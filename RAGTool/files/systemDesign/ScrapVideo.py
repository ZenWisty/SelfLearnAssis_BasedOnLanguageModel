from bilibili_api import video
import requests
import os

async def download_audio(video_id):
    v = video.Video(bvid=video_id)
    info = await v.get_info()
    audio_url = None
    for page in info["pages"]:
        if "audio" in page.get("accept_quality", []):
            cid = page["cid"]
            audio_url = f"https://api.bilibili.com/x/player/playurl?cid={cid}&bvid={video_id}&qn=0&type=audio"
            break
    if audio_url:
        response = requests.get(audio_url)
        output_folder = r"E:\Python_work\LLM_MetaGPT\Publish\SelfLearnAssis_BasedOnLanguageModel\RAGTool\files\systemDesign\audioData"
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        with open(os.path.join(output_folder, "audio.mp3"), "wb") as f:
            f.write(response.content)
            print("音频下载成功")
    else:
        print("未找到音频资源")

import asyncio
video_id = "BV19mkbYQEWM"
asyncio.run(download_audio(video_id))
