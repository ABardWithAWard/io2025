# Import necessary libraries
import os  # For file path operations
import torch  # Main deep learning framework
from torch.utils.data import Dataset, DataLoader  # For data loading utilities
from PIL import Image  # For image handling
from torchvision import transforms  # For image transformations
from torch import nn, optim  # Neural network modules and optimizers
from torchvision import models  # Pretrained computer vision models
from torch.nn import TransformerDecoder, TransformerDecoderLayer  # Transformer components


class CharTokenizer:
    """Handles text tokenization/encoding for OCR tasks"""

    def __init__(self):
        import string
        # Define character set: letters, digits, punctuation and space
        # Can expand to polish characters in near future
        chars = list(string.ascii_letters + string.digits + string.punctuation + ' ')

        # Special tokens for sequence processing
        # https://datascience.stackexchange.com/questions/26947/why-do-we-need-to-add-start-s-end-s-symbols-when-using-recurrent-neural-n
        self.pad_token = "[PAD]"  # Padding token for batch processing
        self.sos_token = "[SOS]"  # Start-of-sequence token
        self.eos_token = "[EOS]"  # End-of-sequence token
        # Can possibly add unknown character which should filter out noise??

        # Create vocabulary with special tokens first
        self.vocab = [self.pad_token, self.sos_token, self.eos_token] + chars

        # Create mapping between characters and indices
        self.char2idx = {c: i for i, c in enumerate(self.vocab)}
        self.idx2char = {i: c for c, i in self.char2idx.items()}

    def encode(self, text: str):
        """Convert text string to sequence of token indices"""
        return [self.char2idx[self.sos_token]] + [  # Start with SOS
            self.char2idx[c] for c in text if c in self.char2idx  # Convert valid chars
        ] + [self.char2idx[self.eos_token]]  # End with EOS

    def decode(self, indices):
        """Convert token indices back to text string"""
        # Filter out padding tokens and convert to characters
        chars = [self.idx2char[i] for i in indices if i not in (self.char2idx[self.pad_token],)]
        # Join characters and remove special tokens
        return ''.join(chars).replace(self.sos_token, '').replace(self.eos_token, '')

    def vocab_size(self):
        """Total number of tokens in vocabulary"""
        return len(self.vocab)


class OCRDataset(Dataset):
    """Custom dataset for loading OCR images and labels"""

    def __init__(self, txt_file: str, img_root: str, tokenizer: CharTokenizer, img_size=(128, 32)):
        # Load image paths and labels from text file
        with open(txt_file, 'r') as f:
            self.samples = [line.strip().split('\t') for line in f if line.strip()]

        self.img_root = img_root  # Root directory for images
        self.tokenizer = tokenizer  # Text tokenizer

        # Image preprocessing pipeline:
        self.transform = transforms.Compose([
            transforms.Resize(img_size),  # Resize to fixed dimensions
            transforms.ToTensor(),  # Convert to tensor [0,1] range
            transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1,1] range
            # https://discuss.pytorch.org/t/understanding-transform-normalize/21730
            # Current normalization is good for grayscale images
            # I am far from rgb, but this is usage for rgb:
            #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        """Total number of samples in dataset"""
        return len(self.samples)

    def __getitem__(self, idx):
        """Load and process single sample"""
        rel_path, label = self.samples[idx]  # Get relative path and text label
        img_path = os.path.join(self.img_root, rel_path)  # Full image path

        # Load and preprocess image
        image = self.transform(Image.open(img_path).convert('RGB'))

        # Convert text label to token indices
        token_ids = self.tokenizer.encode(label)

        return image, torch.tensor(token_ids, dtype=torch.long)


def collate_fn(batch):
    """Custom batch processing function for DataLoader"""
    # Separate images and targets
    images, targets = zip(*batch)

    # Stack images into batch tensor [batch_size, channels, height, width]
    images = torch.stack(images)

    # Calculate original sequence lengths
    lengths = [len(t) for t in targets]
    max_len = max(lengths)  # Find longest sequence

    # Create padded tensor filled with padding tokens
    # As far as I understand it pads like hashing algorithms
    padded = torch.full((len(targets), max_len), fill_value=0, dtype=torch.long)

    # Fill padded tensor with actual sequences
    # Without this we have only zeroes
    for i, t in enumerate(targets):
        padded[i, :len(t)] = t

    return images, padded, lengths


class CNNBackbone(nn.Module):
    """CNN feature extractor for image processing"""

    def __init__(self, output_dim: int):
        super().__init__()
        # Use ResNet18 without trained weights
        base_model = models.resnet18(weights=None)
        modules = list(base_model.children())[:-2]  # Remove avgpool and fc layers

        # Create feature extraction backbone
        self.backbone = nn.Sequential(*modules)

        # 1x1 convolution to adjust channel dimension
        # https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
        # kernel size should be 1, 3, 5 or 7 according to
        # https://pyimagesearch.com/2018/12/31/keras-conv2d-and-convolutional-layers/
        self.conv1x1 = nn.Conv2d(512, output_dim, kernel_size=1)

    def forward(self, images):
        # Handle grayscale images by repeating across channels
        if images.shape[1] == 1:
            images = images.repeat(1, 3, 1, 1)

        # Extract features [batch_size, 512, h, w]
        feats = self.backbone(images)

        # Adjust channels [batch_size, output_dim, h, w]
        feats = self.conv1x1(feats)

        # Reshape to sequence format [batch_size, num_patches, output_dim]
        b, c, h, w = feats.shape
        return feats.permute(0, 2, 3, 1).reshape(b, h * w, c)


# https://github.com/vlomme/OCR-transformer/blob/main/ocr.py
# Something like this but altered to make it work in my code
# Black magic
class OCRTransformer(nn.Module):
    """Main OCR model combining CNN and Transformer"""

    def __init__(self, vocab_size: int, embed_dim: int = 256, num_heads: int = 4, num_layers: int = 3):
        super().__init__()
        # Image feature extractor
        self.cnn = CNNBackbone(embed_dim)

        # Text embedding layer
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)

        # Learnable positional embeddings
        self.pos_embedding = nn.Parameter(torch.randn(1, 512, embed_dim))

        # Transformer decoder setup
        decoder_layer = TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads  # Number of attention heads
        )
        self.transformer_decoder = TransformerDecoder(decoder_layer, num_layers=num_layers)

        # Final classification layer
        self.fc_out = nn.Linear(embed_dim, vocab_size)

    def forward(self, images, tgt_input):
        # Extract visual features [batch_size, num_patches, embed_dim]
        visual_feats = self.cnn(images)

        # Create token embeddings + positional embeddings
        tgt_emb = self.token_embedding(tgt_input) + self.pos_embedding[:, :tgt_input.size(1)]

        # Adjust dimensions for Transformer:
        # Transformer expects [seq_len, batch_size, embed_dim]
        visual_feats = visual_feats.permute(1, 0, 2)  # Memory (image features)
        tgt_emb = tgt_emb.permute(1, 0, 2)  # Target sequence

        # Create causal mask to prevent looking ahead
        sz = tgt_input.size(1)
        causal_mask = torch.triu(torch.ones(sz, sz, device=tgt_input.device), diagonal=1).bool()

        # Transformer decoding process
        out = self.transformer_decoder(
            tgt=tgt_emb,
            memory=visual_feats,
            tgt_mask=causal_mask
        )

        # Convert decoder output to vocabulary probabilities
        logits = self.fc_out(out.permute(1, 0, 2))  # [batch_size, seq_len, vocab_size]
        return logits


def train_model():
    """Training loop for OCR model"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize data components
    tokenizer = CharTokenizer()
    dataset = OCRDataset(
        txt_file=r"...",  # Path to training annotations
        img_root=r"...",  # Path to image directory
        tokenizer=tokenizer
    )
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)

    # Initialize model and optimizer
    model = OCRTransformer(vocab_size=tokenizer.vocab_size()).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Loss function ignoring padding tokens
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.char2idx[tokenizer.pad_token])

    # Training loop
    for epoch in range(100):
        model.train()
        total_loss = 0

        for images, targets, lengths in dataloader:
            # Move data to device
            images, targets = images.to(device), targets.to(device)

            # Create shifted version for teacher forcing
            tgt_input = targets[:, :-1]  # Input to decoder
            tgt_output = targets[:, 1:]  # Expected output

            # Forward pass
            logits = model(images, tgt_input)

            # Reshape for loss calculation
            logits = logits.reshape(-1, logits.shape[-1])  # [batch*seq_len, vocab]
            tgt_output = tgt_output.reshape(-1)  # [batch*seq_len]

            # Calculate loss
            loss = loss_fn(logits, tgt_output)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # Print epoch statistics
        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(dataloader):.4f}")

        # Save model checkpoint
        save_path = r"..."
        torch.save(model.state_dict(), save_path)
        print(f"Model saved to {save_path}")


def infer_image(filepath: str, model_path: str = "...") -> str:
    """Perform inference on a single image"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device")

    # Initialize components
    tokenizer = CharTokenizer()
    model = OCRTransformer(vocab_size=tokenizer.vocab_size())
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device).eval()

    # Image preprocessing pipeline
    transform = transforms.Compose([
        transforms.Resize((128, 32)),  # Match training size
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Load and preprocess image
    image = Image.open(filepath).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)  # Add batch dimension

    # Initialize generation with start token
    input_seq = torch.tensor([[tokenizer.char2idx[tokenizer.sos_token]]], dtype=torch.long).to(device)

    # Autoregressive generation loop
    for _ in range(100):  # Maximum sequence length
        # Get predictions
        logits = model(image_tensor, input_seq)

        # Greedy decoding: select most probable next token
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)

        # Append to input sequence
        input_seq = torch.cat([input_seq, next_token], dim=1)

        # Stop if end token generated
        if next_token.item() == tokenizer.char2idx[tokenizer.eos_token]:
            break

    # Convert indices to text
    decoded_text = tokenizer.decode(input_seq[0].tolist())
    return decoded_text.strip()  # Remove whitespace