import os
import subprocess
from google.colab import drive
from dotenv import load_dotenv

def mount_drive():
    drive.mount('/content/gdrive')

def install_requirements():
    subprocess.run(['pip', 'install', '-r', 'requirements.txt'])

def reload_environment():
    try:
        from IPython import get_ipython
        ipython = get_ipython()
        ipython.kernel.do_shutdown(restart=True)
        print("Kernel restarted. Packages should be reloaded.")
    except Exception as e:
        print(f"Kernel restart failed: {e}")
        print("Consider manually restarting the kernel or your Jupyter Notebook server.")

def setup_environment():
    # Monter Google Drive
    mount_drive()

    # Charger les variables d'environnement
    load_dotenv()
    secret_hf = os.getenv('HF_TOKEN')

    if not secret_hf:
        print("Please go to the following URL and obtain your Hugging Face token:")
        print("https://huggingface.co/settings/tokens")
        print()
        hf_token = input("Please enter your Hugging Face token: ")

        with open('.env', 'a') as f:
            f.write(f"HF_TOKEN={hf_token}\n")

    # Connexion à Hugging Face
    subprocess.run(['huggingface-cli', 'login', '--token', secret_hf])

    # Installation des dépendances
    install_requirements()

if __name__ == "__main__":
    setup_environment()
    reload_environment()
    print("Setup complete.")
