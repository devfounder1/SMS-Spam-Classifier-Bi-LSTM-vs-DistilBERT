from huggingface_hub import snapshot_download
import os

print("Скачивание моделей с Hugging Face Hub")

# Скачиваем весь репозиторий в папку /app/models
snapshot_download(
    repo_id="butuzik/sms-spam-models",
    local_dir="/app/models",
    local_dir_use_symlinks=False
)

print("Модели успешно скачаны!")
print("Структура папки models:")
for root, dirs, files in os.walk("/app/models"):
    level = root.replace("/app/models", "").count(os.sep)
    indent = " " * 2 * level
    print(f"{indent}{os.path.basename(root)}/")
    subindent = " " * 2 * (level + 1)
    for file in files:
        filepath = os.path.join(root, file)
        size = os.path.getsize(filepath)
        print(f"{subindent}{file} ({size / 1024 / 1024:.2f} MB)")