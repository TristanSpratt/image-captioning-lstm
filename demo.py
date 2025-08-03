import os
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from image_captioning import (
    get_resnet18_encoder,
    encode_images,
    load_image_list,
    load_captions,
    build_vocab,
    CaptionDataset,
    CaptionGenerator,
    train_one_epoch,
    greedy_decode,
    load_image
)

def main():
    # ----------------------
    # CONFIGURATION
    # ----------------------
    DATA_DIR = "data/flickr8k"
    IMG_DIR = os.path.join(DATA_DIR, "Flickr8k_Dataset")
    TRAIN_LIST = os.path.join(DATA_DIR, "Flickr_8k.trainImages.txt")
    CAPTIONS_FILE = os.path.join(DATA_DIR, "Flickr8k.token.txt")
    OUTPUT_DIR = "outputs/captions/"
    N_EPOCHS = 3
    LR = 1e-3
    SEED = 42
    EXAMPLES = 3

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    random.seed(SEED)
    torch.manual_seed(SEED)

    # ----------------------
    # LOAD DATA
    # ----------------------
    print("Loading data and captions...")
    train_images = load_image_list(TRAIN_LIST)
    descriptions = load_captions(CAPTIONS_FILE)
    word_to_id, id_to_word = build_vocab(descriptions)

    # ----------------------
    # ENCODE IMAGES
    # ----------------------
    print("Encoding images with ResNet18...")
    encoder = get_resnet18_encoder()
    subset = train_images[:100]
    img_encodings = encode_images(subset, IMG_DIR, encoder)

    # ----------------------
    # BUILD DATASET & MODEL
    # ----------------------
    print("Building dataset and model...")
    dataset = CaptionDataset(subset, img_encodings, descriptions, word_to_id)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)
    model = CaptionGenerator(len(word_to_id) + 1).to(torch.device("cpu"))

    # ----------------------
    # TRAIN
    # ----------------------
    print("Training model...")
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    for epoch in range(N_EPOCHS):
        loss, acc = train_one_epoch(model, loader, optimizer, loss_fn)
        print(f"Epoch {epoch+1}: Loss={loss:.4f}, Accuracy={acc:.2f}")

    # ----------------------
    # INFERENCE & VISUALIZATION
    # ----------------------
    print(f"\nGenerating captions for {EXAMPLES} images...")
    sample_imgs = random.sample(subset, EXAMPLES)
    fig, axes = plt.subplots(1, EXAMPLES, figsize=(5 * EXAMPLES, 5))

    for idx, name in enumerate(sample_imgs):
        path = os.path.join(IMG_DIR, name)
        caption = greedy_decode(model, encoder, path, word_to_id, id_to_word)
        image_raw = np.array(Image.open(path).convert('RGB')) / 255.0

        axes[idx].imshow(image_raw)
        axes[idx].set_title(" ".join(caption), fontsize=8)
        axes[idx].axis("off")

        image_tensor = load_image(path)
        save_image(image_tensor, os.path.join(OUTPUT_DIR, f"{name}_captioned.png"))

    plt.tight_layout()
    grid_path = os.path.join(OUTPUT_DIR, "demo_grid.png")
    plt.savefig(grid_path)
    print(f"\nSaved caption grid to {grid_path}")
    plt.show()

if __name__ == "__main__":
    main()
