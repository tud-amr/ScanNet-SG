import requests
import os
from openai import OpenAI
import base64
import argparse
import numpy as np
import cv2

class Client:
    def __init__(self, cfgs):
      self.url = cfgs.url
      self.prompt = cfgs.prompt
      self.image = cfgs.image
      self.mode = cfgs.mode

      # Initialize the OpenAI client
      self.gpt_client = OpenAI(
          api_key=os.environ.get("OPENAI_API_KEY"),
      )

    # Function to encode the image
    def encode_image(self, image_path):
      with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

    # Function to load the image
    def load_image(self, image_path):
      with open(image_path, "rb") as f:
        image_data = f.read()
      return image_data
    
    def gpt_inference(self):
      if self.mode == "main":
        # Load the image
        image_path = self.image
        self.image_data = self.load_image(image_path)
        base64_image = self.encode_image(image_path)
      else:
        self.image_data = cv2.imencode('.jpg', self.image)[1]
        base64_image = base64.b64encode(self.image_data).decode('utf-8')

      # Prompts
      prompts = self.prompt

      response = self.gpt_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
          {
            "role": "user",
            "content": [
              {
                "type": "text",
                "text": f"Here is a user question: {prompts}. If it asks if one thing is in the image, just answer yes or no. However if the answer is yes, please also attach (only) the name the the object, with comma but without full stop \
                          If it asks what is/are in the image, only reply the names of objects in the image, without full stop. \
                          For any other questions beyond these two, reply that the question type is not supported.",  #didn't work
              },
              {
                "type": "image_url",
                "image_url": {
                  "url":  f"data:image/jpeg;base64,{base64_image}"
                },
              },
            ],
          }
        ],
      )

      response_message = response.choices[0].message.content
      result = response_message.split(", ")
      print(response_message)
      if "No" in result:
          raise SystemExit("The object is not in the image")
      elif "Yes" in result:
        self.class_name = result[1]
      elif "not supported" in result:
          print("The question type is not supported") 
      else:
        self.class_name = result

    def send_request(self):
      self.gpt_inference()
      print(f'class_name: {self.class_name}')
      # Send request
      response = requests.post(
          self.url,
          files={"image": ("example.png", self.image_data, "image/png")},
          data={"prompts": self.class_name}
      )

      # Get the text and image response
      if response.status_code == 200:
          data = response.json()

          decoded_image_data = base64.b64decode(data['image'])
          nparr = np.frombuffer(decoded_image_data, np.uint8)
          image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)


          decoded_masks_data = [base64.b64decode(mask) for mask in data['masks']]
          masks = [cv2.imdecode(np.frombuffer(mask, np.uint8), cv2.IMREAD_COLOR) for mask in decoded_masks_data]

          class_names = data['classes']

          if self.mode == "main":
            # Save the output image
            cv2.imwrite("result/output.jpg", image)
            for i, mask in enumerate(masks):
                cv2.imwrite(f"result/{class_names[i]}_{i}.jpg", mask)
            print("Segmentation finished. Check the output.jpg and output_mask.jpg files.")
          else:
            return image, masks, class_names
      else:
          print("Error:", response.json())

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--prompt', required=True, help='Ask a question about the image')
  # parser.add_argument('--url', help='Server URL', default="http://145.94.60.29:8000/process")
  parser.add_argument('--url', help='Server URL', default="http://localhost:5000/process")
  parser.add_argument('--image', help='image you want to process', default="/home/cc/Pictures/kitchen_scene.png")
  parser.add_argument('--mode', help='Mode of the client. Choose from \'main\' and \'ros\'', default="main")
  cfgs = parser.parse_args()
  client = Client(cfgs)
  client.send_request()
