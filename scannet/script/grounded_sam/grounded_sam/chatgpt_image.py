import os
from openai import OpenAI
import base64
import requests


# Initialize the OpenAI client
client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
)

# Function to encode the image
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')
  

# Path to your image file
image_path = "/home/cc/chg_ws/ros_ws/topomap_ws/src/data/test/images/scans/scene0702_00/frame-000111.color.jpg"
base64_image = encode_image(image_path)

response = client.chat.completions.create(
  model="gpt-4o-mini",
  messages=[
    {
      "role": "user",
      "content": [
        {
          "type": "text",
          "text": "What are the objects in this image (exclude floor, wall, ceiling, etc.)? Give the name and a detailed description each object and present in a json format.",
        },
        {
          "type": "image_url",
          "image_url": {
            "url": f"data:image/jpeg;base64,{base64_image}"
          },
        },
      ],
    }
  ],
)

print(response.choices[0].message.content)
