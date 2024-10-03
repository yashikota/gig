import requests

def send_audio_to_api(file_path, url):
    filetype = file_path.split('.')[-1]

    with open(file_path, 'rb') as f:
        files = {'file': (file_path, f, f'audio/{filetype}')}
        response = requests.post(url, files=files)
    return response

api_url = "http://127.0.0.1:8088/transcribe"
wav_file_path = r"testdata/digital.mp3"
response = send_audio_to_api(wav_file_path, api_url)

if response.status_code == response.ok:
    print("Success:", response.json())
else:
    print("Error:", response.text)
