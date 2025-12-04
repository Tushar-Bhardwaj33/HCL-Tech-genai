import getpass
import os
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langchain_core.embeddings import Embeddings
import os
import sys
import uuid
import base64
import requests
import json
import numpy as np
from pathlib import Path
import os
from dotenv import load_dotenv

# Load the .env file
load_dotenv()

__all__ = ["embeddings"]

class embeddings:
    def __init__(self):
        pass

    def get_availbale_methods(self):
        return [self.embed_query, self.embed_documents, self.embed_image]

    def embed_query(self, text):
        if not os.environ.get("NVIDIA_API_KEY"):
          os.environ["NVIDIA_API_KEY"] = getpass.getpass("Enter API key for NVIDIA: ")

        embeddings = NVIDIAEmbeddings(model="NV-Embed-QA")
        response = embeddings.embed_query(text)
        return response
    
    def embed_documents(self, documents: list):
        if not os.environ.get("NVIDIA_API_KEY"):
            os.environ["NVIDIA_API_KEY"] = getpass.getpass("Enter API key for NVIDIA: ")

        embedder = NVIDIAEmbeddings(model="NV-Embed-QA")
        d_embeddings = embedder.embed_documents(
            documents=documents
        )
        return d_embeddings

    def embed_image(self, prompt: str = None, image_b64: str = None, image_file: str = None, api_key: str = None):
      if image_b64 is None:
          with open(image_file, "rb") as image_file:
              image_b64 = base64.b64encode(image_file.read()).decode("utf-8")
      if len(image_b64) > 200000:
        raise Exception("Image size too large")
      nvai_url="https://ai.api.nvidia.com/v1/cv/nvidia/nv-dinov2"
      if not os.environ.get("NVIDIA_API_KEY"):
          os.environ["NVIDIA_API_KEY"] = getpass.getpass("Enter API key for NVIDIA: ")
      header_auth = f"Bearer {os.getenv('NVIDIA_API_KEY')}"
      payload = {
        "messages": [
          {
            "content": {
                "type": "image_url",
                "image_url": {
                  "url": f"data:image/jpeg;base64,{image_b64}"
                }
            }
          }
        ],
      }

      headers = {
          "Content-Type": "application/json",
          "Authorization": header_auth,
          "Accept": "application/json"
      }
      response = requests.post(nvai_url, headers=headers, json=payload)

      if response.status_code == 200:
          return response.json()
      else:
          print('Inference failed.')