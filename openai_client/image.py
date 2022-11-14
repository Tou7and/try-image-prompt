import os
import subprocess
import openai

with open("exp/mykey.txt", 'r') as reader:
    openai.api_key = reader.read().strip()

# prompt_string = "worker intellengence"
# prompt_string = "Why Don't You Ask The Magic Conch"
# prompt_string = "SpongeBob Asking The Magic Conch"
# prompt_string = "SpongeBob and a large purple conch"
prompt_string = "SpongeBob Jojo Stand"

response = openai.Image.create(
    prompt=prompt_string,
    n=3,
    size="1024x1024"
)

print(response)
data = response["data"]

for idx, datum in enumerate(data):
    url = datum['url']
    filename = f"exp/img-{idx}.png"
    print(idx)
    # use wget to download image from url
    subprocess.run(["wget", "-O", filename, url])
