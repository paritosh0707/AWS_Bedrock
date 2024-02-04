import json
import boto3
import base64
import os

prompt = """
provide me a 4K image of a beach, also use a blue sky rainy season and cinematic display
"""

prompt_template = {"text":prompt}
img_config = {
        "cfgScale":8,
        "seed":0,
        "quality":"standard",
        "width":1024,
        "height":1024,
        "numberOfImages":1
    }
bedrock = boto3.client(service_name="bedrock-runtime")

payload = {
    "textToImageParams":prompt_template,
    "taskType":"TEXT_IMAGE",
    "imageGenerationConfig":img_config
}

body = json.dumps(payload)
model_id = 'amazon.titan-image-generator-v1'
response = bedrock.invoke_model(
    body=body,
    modelId=model_id,
    accept="application/json",
    contentType="application/json"
)

response_body = json.loads(response.get("body").read())
artifact = response_body["images"][0]
image_encoded = artifact.encode("utf-8")
image_bytes = base64.b64decode(image_encoded)

output_directory= "output"
os.makedirs(output_directory, exist_ok=True)

final_name = f"{output_directory}/img.png"
with open(final_name, 'wb') as f:
    f.write(image_bytes)