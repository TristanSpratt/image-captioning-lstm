import os
import torch
import numpy as np
from PIL import Image
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchvision.models import ResNet18_Weights

# ----------------------
# DEVICE CONFIGURATION
# ----------------------
# Default to CPU for use on macOS (problems with pytorch and mps)
DEVICE = torch.device("cpu")

# Uncomment below to use CUDA if available
# DEVICE = (
#     torch.device("cuda") if torch.cuda.is_available()
#     else torch.device("mps") if torch.backends.mps.is_available()
#     else torch.device("cpu")
# )

# ----------------------
# GLOBAL CONFIGS
# ----------------------
MAX_LEN = 40
EMBED_DIM = 512
HIDDEN_DIM = 512
BATCH_SIZE = 16

# ----------------------
# IMAGE PREPROCESSING
# ----------------------
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def load_image(path):
    image = Image.open(path).convert('RGB')
    return preprocess(image)

# ----------------------
# DATA UTILS
# ----------------------
def load_image_list(filename):
    with open(filename, 'r') as f:
        return [line.strip() for line in f]

def load_captions(filename):
    """Returns {image_id: [caption_tokens, ...]}"""
    desc = {}
    with open(filename, 'r') as f:
        for line in f:
            key, caption = line.strip().split('\t')
            key = key.split('#')[0]
            words = ['<START>'] + caption.lower().split() + ['<EOS>']
            if key not in desc:
                desc[key] = []
            desc[key].append(words)
    return desc

def build_vocab(descriptions):
    token_set = set()
    for caps in descriptions.values():
        for cap in caps:
            token_set.update(cap)
    token_set -= {'<PAD>', '<START>', '<EOS>'}
    ordered = sorted(token_set)
    id_to_word = {0: '<PAD>', 1: '<START>', 2: '<EOS>'}
    word_to_id = {'<PAD>': 0, '<START>': 1, '<EOS>': 2}
    for i, word in enumerate(ordered):
        id_to_word[i + 3] = word
        word_to_id[word] = i + 3
    return word_to_id, id_to_word

# ----------------------
# IMAGE ENCODER
# ----------------------
def get_resnet18_encoder():
    model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    modules = list(model.children())[:-1]  # remove final fc layer
    encoder = nn.Sequential(*modules)
    return encoder.to(DEVICE).eval()

def encode_images(image_list, image_dir, encoder):
    tensor_out = torch.zeros((len(image_list), 512), device=DEVICE)
    with torch.no_grad():
        for i, name in enumerate(image_list):
            image = load_image(os.path.join(image_dir, name))
            encoded = encoder(image.unsqueeze(0).to(DEVICE))[:, :, 0, 0]  # [1, 512, 1, 1] -> [1, 512]
            tensor_out[i] = encoded.squeeze(0)
    return tensor_out

# ----------------------
# DATASET
# ----------------------
class CaptionDataset(Dataset):
    def __init__(self, img_list, img_encodings, descriptions, word_to_id):
        self.data = []
        self.img_enc = img_encodings
        self.img_index = {name: i for i, name in enumerate(img_list)}
        self.w2id = word_to_id

        for name in img_list:
            if name not in descriptions: continue
            img_vec = self.img_enc[self.img_index[name]]
            for cap in descriptions[name]:
                enc = [self.w2id[word] for word in cap]
                for i in range(1, len(enc)):
                    x = enc[:i] + [self.w2id['<PAD>']] * (MAX_LEN - i)
                    y = enc[1:i+1] + [self.w2id['<PAD>']] * (MAX_LEN - i)
                    self.data.append((img_vec, x, y))

    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
        img, x, y = self.data[idx]
        return img, torch.tensor(x), torch.tensor(y)

# ----------------------
# MODEL
# ----------------------
class CaptionGenerator(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, EMBED_DIM)
        self.lstm = nn.LSTM(EMBED_DIM + 512, HIDDEN_DIM, batch_first=True)
        self.output = nn.Linear(HIDDEN_DIM, vocab_size)

    def forward(self, img_enc, seq):
        # On MPS (Mac GPU), torch.nn.Embedding and LSTM 
        # can fail due to memory allocation issues.
        # This version keeps the embedding on CPU if necessary, 
        # while allowing GPU execution if supported.
        embed = self.embedding(seq.cpu())
        embed = embed.to(img_enc.device)
        img_expanded = img_enc.unsqueeze(1).expand(-1, embed.size(1), -1)
        x = torch.cat((embed, img_expanded), dim=2)
        lstm_out, _ = self.lstm(x)
        return self.output(lstm_out)

    # Uncomment below to use GPU if available
    # def forward(self, img_enc, seq):
    #     embed = self.embedding(seq)
    #     img_expanded = img_enc.unsqueeze(1).expand(-1, embed.size(1), -1)
    #     x = torch.cat((embed, img_expanded), dim=2)
    #     lstm_out, _ = self.lstm(x)
    #     return self.output(lstm_out)

# ----------------------
# TRAINING
# ----------------------
def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss, total_correct, total_preds = 0, 0, 0
    for img, x, y in loader:
        img, x, y = img.to(DEVICE), x.to(DEVICE), y.to(DEVICE)
        logits = model(img, x)
        loss = criterion(logits.transpose(2, 1), y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        preds = torch.argmax(logits, dim=2)
        mask = y != 0
        total_correct += ((preds == y) & mask).sum().item()
        total_preds += mask.sum().item()
        total_loss += loss.item()
    
    acc = total_correct / total_preds if total_preds > 0 else 0
    return total_loss / len(loader), acc

# ----------------------
# INFERENCE
# ----------------------
def greedy_decode(model, encoder, image_path, word_to_id, id_to_word):
    model.eval()
    image = load_image(image_path).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        img_vec = encoder(image)[:, :, 0, 0]  # [1, 512]
        seq = [word_to_id['<START>']] + [word_to_id['<PAD>']] * (MAX_LEN - 1)
        seq_tensor = torch.tensor(seq).unsqueeze(0).to(DEVICE)
        generated = ['<START>']
        for i in range(MAX_LEN - 1):
            logits = model(img_vec, seq_tensor)
            next_id = torch.argmax(logits[0, i]).item()
            next_word = id_to_word[next_id]
            generated.append(next_word)
            if next_word == '<EOS>': break
            seq_tensor[0, i + 1] = next_id
    return generated
