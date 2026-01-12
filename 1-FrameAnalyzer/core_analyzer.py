import os
import cv2
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import base64
import json
import re
import logging
from datetime import datetime, timedelta
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI


load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(), logging.FileHandler("analysis.log", encoding='utf-8')]
)
logger = logging.getLogger(__name__)

class BaseVideoAnalyzer:
    def __init__(self, model_name="qwen2.5-vl-72b-instruct"):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        
        resnet_base = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.feature_model = nn.Sequential(*list(resnet_base.children())[:-1]).to(self.device).eval()
        self.resnet_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
       
        api_key = os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            raise ValueError("Environment variable DASHSCOPE_API_KEY is missing!")

        self.llm = ChatOpenAI(
            model=model_name,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            api_key=api_key,
        )
        self._reader = None

    @property
    def reader(self):
        if self._reader is None:
            import easyocr
            self._reader = easyocr.Reader(['ch_sim', 'en'], gpu=(self.device == 'cuda'))
        return self._reader

    def get_frame_feature(self, frame):
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_t = self.resnet_transform(Image.fromarray(img_rgb)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return self.feature_model(img_t).flatten()

    def call_llm(self, prompt, frames_data):
        contents = [{"type": "text", "text": prompt}]
        for f in frames_data:
            _, buffer = cv2.imencode('.jpg', f['frame'], [cv2.IMWRITE_JPEG_QUALITY, 75])
            img_b64 = base64.b64encode(buffer).decode('utf-8')
            contents.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}
            })
        
        try:
            resp = self.llm.invoke([HumanMessage(content=contents)])
            clean_text = re.sub(r'```json\n?|```', '', resp.content).strip()
            match = re.search(r'\{.*\}', clean_text, re.DOTALL)
            return json.loads(match.group(0)) if match else json.loads(clean_text)
        except Exception as e:
            logger.error(f"LLM Error: {e}")
            return {"events": []}
        
    def save_report(self, output_dir, filename, data):
        os.makedirs(output_dir, exist_ok=True)
        report_file = os.path.join(output_dir, filename)
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        logger.info(f"✨ 任务完成！分析报告已保存至: {report_file}")