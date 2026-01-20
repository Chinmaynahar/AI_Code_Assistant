import torch
from ui.app import launch_app
from config.settings import DEVICE


def main():
    print("🚀 Starting RAG Code Assistant with Smart Crawling...")
    print(f"📊 Device: {DEVICE.upper()}")
    
    if torch.cuda.is_available():
        print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
    
    print("\n🌐 Opening web interface...")
    print("\n💡 TIP: Use 'Recursive Crawler' mode to automatically follow all documentation links!")
    
    launch_app()


if __name__ == "__main__":
    main()