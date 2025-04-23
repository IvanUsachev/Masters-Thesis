import gdown
import os


os.makedirs("models", exist_ok=True)

file_id = "1u9PA9ZwHD9DJnUOu_nr3__4-H_5ASBCk"

output_path = "models/trained_models.zip"

url = f"https://drive.google.com/uc?id={file_id}"

gdown.download(url, output_path, quiet=False)