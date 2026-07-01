from src.data_collection.collector import collect_all


def main():
    print("=" * 50)
    print("Downloading maritime law data from public sources...")
    print("=" * 50)
    texts = collect_all()
    print(f"\nDownloaded {len(texts)} documents:")
    for name, text in texts.items():
        print(f"  - {name}: {len(text)} characters")
    print("\nDone!")


if __name__ == "__main__":
    main()
