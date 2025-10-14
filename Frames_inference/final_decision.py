from llm.api_video_inference import api_inference_video
from prompt.prompt import overall_prompt
from llm.local_video_inference import local_inference_video

def result(result, model_name, use_api):
    # 调用其他大模型api
    #model_name = "qwen2.5-vl-32b-instruct"


    all_replies = ' \n'.join(result)
    prompt = overall_prompt + all_replies
    if use_api:
        res = api_inference_video(model_name, prompt)
    else:
        res = local_inference_video(model_name, prompt)

    print(res)
    return res


