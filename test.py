import requests

# Put any random .jpg image in your folder and change this name!
image_filename = "test_image.jpg" 

url = "http://127.0.0.1:8000/predict"

print(f"Sending {image_filename} to the FastAPI server...")
with open(image_filename, "rb") as f:
    response = requests.post(url, files={"file": f})

if response.status_code == 200:
    with open("result_mask.png", "wb") as f:
        f.write(response.content)
    print("Success! Saved the predicted mask as 'result_mask.png'.")
else:
    print(f"Failed. Server returned status code: {response.status_code}")