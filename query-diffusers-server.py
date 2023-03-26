import requests

data = {
    "prompt": "photo, a church in the middle of a field of crops, bright cinematic lighting, gopro, fisheye lens"
}

response = requests.post("http://localhost:5000/generate", data=data)

# Save the image to a file
with open('result.jpg', 'wb') as f:
    f.write(response.content)