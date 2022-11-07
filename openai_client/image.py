import os
import openai

with open("private/mykey.txt", 'r') as reader:
    openai.api_key = reader.read().strip()

prompt_string = "worker intellengence"

response = openai.Image.create(
  prompt=prompt_string,
  n=2,
  size="1024x1024"
)

print(response)
