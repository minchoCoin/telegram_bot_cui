# telegram_bot_cui
talk to your telegram bot with Conversational User Interface(CUI), on-device TTS and STT

# requirements

## turn on local LLM
In my case, I adopt the gpt-oss-120b on DGX Spark

thanks to [https://github.com/christopherowen/spark-vllm-mxfp4-docker/](https://github.com/christopherowen/spark-vllm-mxfp4-docker/), we can run gpt-oss with MXFP4 on DGX-SPARK.

``` bash
git clone https://github.com/christopherowen/spark-vllm-mxfp4-docker.git
cd spark-vllm-mxfp4-docker
docker build -t vllm-mxfp4-spark .
hf download openai/gpt-oss-120b
docker compose up -d
```

## run openclaw
```
curl -fsSL https://openclaw.ai/install.sh | bash
```
```
Model/auth provider: vllm
API Base URL: http://127.0.0.1:8000
API Key: custom
model: gpt-oss-120b
Endpoint compatibility: OpenAI-compatible
```
```json
//openclaw.json
"models": {
    "mode": "merge",
    "providers": {
      "vllm": {
        "baseUrl": "http://127.0.0.1:8000/v1",
        "apiKey": "VLLM_API_KEY",
        "api": "openai-completions",
        "models": [
          {
            "id": "gpt-oss-120b",
            "name": "gpt-oss-120b",
            "reasoning": false,
            "input": [
              "text"
            ],
            "cost": {
              "input": 0,
              "output": 0,
              "cacheRead": 0,
              "cacheWrite": 0
            },
            "contextWindow": 131072,
            "maxTokens": 32768
          }
        ]
      },
    }
}
```
### connect telegram

refer the [https://docs.openclaw.ai/channels/telegram](https://docs.openclaw.ai/channels/telegram)

한국 사용자용 추가 가이드: [https://youtu.be/t_YRzdfDzds](https://youtu.be/t_YRzdfDzds)

# Connect telegram and raspberry pi
## Creating telegram API key

refer the [https://core.telegram.org/api/obtaining_api_id](https://core.telegram.org/api/obtaining_api_id)

you can generate the API key on [https://my.telegram.org.](https://my.telegram.org)

## make .env files
```py
api_id=123456
api_hash='your_api_hash'
bot_username='@your_openclaw_bot_id'
```



## install packages
```bash
sudo apt install portaudio19-dev
sudo apt install libportaudio2 libasound-dev
```
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
## login
```
python telegram_login.py
```
# on-device STT
## install whisper.cpp
```bash
git clone https://github.com/ggml-org/whisper.cpp.git
cd whisper.cpp
sh ./models/download-ggml-model.sh small
cmake -B build
cmake --build build -j --config Release
```

# telegram login
```
python telegram_login.py
```

# run with loop and wakeword
```
python main_loop.py
```
# run with loop and without wakeword
```
python main_loop_withoutwake.py
```
# run without loop and with wakeword
```
python main.py
```
# run without loop and wakeword
```
python main_withoutwake.py
```