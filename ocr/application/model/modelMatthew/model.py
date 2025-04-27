import os  # For file path operations
import cv2
import torch  # Main deep learning framework
from torch.utils.data import Dataset, DataLoader  # For data loading utilities
from PIL import Image  # For image handling
from torchvision import transforms  # For image transformations
from torch import nn, optim  # Neural network modules and optimizers
from torchvision import models  # Pretrained computer vision models
from torch.nn import TransformerDecoder, TransformerDecoderLayer  # Transformer components
import string
from application.model.modelMatthew.findingWords import preprocessWords
from application.model.modelMatthew.textSectors import process_images
from application.model.modelbase import ModelBase


DEBUG_MODE = False

class Model(ModelBase):
    """Handles text tokenization/encoding for OCR tasks"""

    def __init__(self):
        super().__init__("OCR")
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

    def perform_ocr(self, input_path, output_path ) -> str:
        """Perform inference on a single image"""
        try:
            model_path = r".\model.pth"
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"Using {device} device")

            # Initialize components
            tokenizer = Model()
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
            image = Image.open(input_path).convert('RGB')
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
        except Exception as e:
            return f"Unexpected error: {e}"

    def _preprocess(self, data_dir):
        """
        Function which detects text sectors and tries to cut them into single lines or words.
        Argument image is path to image which will be preprocessed.
        Does not return anything, but saves cut images to directories created by it.
        """
        try:
            input_dir = os.environ['UPLOADED_FILES']
            imageLoad = cv2.imread(data_dir)
            gray = cv2.cvtColor(imageLoad, cv2.COLOR_BGR2GRAY)
            if DEBUG_MODE:
                cv2.imwrite("UploadedFiles/gray.png", gray)

            blur = cv2.GaussianBlur(gray, (7, 7), 0)
            cv2.imwrite("UploadedFiles/gray_blurred.png", blur)

            thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
            if DEBUG_MODE:
                cv2.imwrite("UploadedFiles/thresh.png", thresh)

            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 13))
            dilate = cv2.dilate(thresh, kernel, iterations=1)
            if DEBUG_MODE:
                cv2.imwrite(f"{input_dir}/dilate.png", dilate)

            # Everything above this line prepares for text sectors detection,
            # we do things like blurring the image, graying it out to reduce noice
            # and then dilate the rest to extract text sectors
            # then we write boxes on text sectors and we splinter original file according to them

            contours = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = contours[0] if len(contours) == 2 else contours[1]

            boxes = [cv2.boundingRect(c) for c in contours if cv2.boundingRect(c)[2] > 1 and cv2.boundingRect(c)[3] > 1]
            img_height = imageLoad.shape[0]
            tolerance = int(0.10 * img_height)

            filtered_boxes = []
            for i, (x1, y1, w1, h1) in enumerate(boxes):
                inside = False
                for j, (x2, y2, w2, h2) in enumerate(boxes):
                    if i != j:
                        if x1 >= x2 and y1 >= y2 and x1 + w1 <= x2 + w2 and y1 + h1 <= y2 + h2:
                            inside = True
                            break
                if not inside:
                    filtered_boxes.append((x1, y1, w1, h1))

            def boxes_are_close(b1, b2, thresh=15):
                # Checks if boxes are close enough to merge them.
                # Threshold is in pixels
                x1, y1, w1, h1 = b1
                x2, y2, w2, h2 = b2
                return not (
                            x1 + w1 + thresh < x2 or x2 + w2 + thresh < x1 or y1 + h1 + thresh < y2 or y2 + h2 + thresh < y1)

            def merge_boxes(b1, b2):
                #Merges close boxes
                x1, y1, w1, h1 = b1
                x2, y2, w2, h2 = b2
                x = min(x1, x2)
                y = min(y1, y2)
                x_max = max(x1 + w1, x2 + w2)
                y_max = max(y1 + h1, y2 + h2)
                return (x, y, x_max - x, y_max - y)

            merged = True
            while merged:
                merged = False
                new_boxes = []
                skip = set()
                for i in range(len(filtered_boxes)):
                    if i in skip:
                        continue
                    box1 = filtered_boxes[i]
                    for j in range(i + 1, len(filtered_boxes)):
                        if j in skip:
                            continue
                        box2 = filtered_boxes[j]
                        if boxes_are_close(box1, box2):
                            box1 = merge_boxes(box1, box2)
                            skip.add(j)
                            merged = True
                    new_boxes.append(box1)
                filtered_boxes = new_boxes

            def sort_key(box):
                #Sorts boxes vertically and horizontally. We sort vertically according to a tolerance,
                #as curved text skewes results
                return (box[1] // tolerance, box[0])

            sorted_boxes = sorted(filtered_boxes, key=sort_key)

            img_w, img_h = imageLoad.shape[1], imageLoad.shape[0]

            def is_horizontal_line(box):
                #Checks if we detected a divider as text.
                #Reduces noise overall when document is partitioned
                x, y, w, h = box
                aspect_ratio = w / h if h > 0 else 0
                return h <= 15 and aspect_ratio > 10

            #We filter out very small boxes which is likely noise
            #w and h can be adjusted to smaller/bigger values
            final_boxes = [(x, y, w, h) for (x, y, w, h) in sorted_boxes if w >= 25 and h >= 25]

            # Split boxes containing horizontal lines
            line_boxes = [box for box in final_boxes if is_horizontal_line(box)]
            other_boxes = [box for box in final_boxes if not is_horizontal_line(box)]
            used_lines = []
            new_other_boxes = []

            for other_box in other_boxes:
                #Splintering file according to boxes
                ox, oy, ow, oh = other_box
                split_lines = []
                for line_box in line_boxes:
                    lx, ly, lw, lh = line_box
                    if (lx >= ox and ly >= oy and (lx + lw) <= (ox + ow) and (ly + lh) <= (oy + oh)):
                        if lw >= 0.8 * ow:
                            split_lines.append(line_box)
                            used_lines.append(line_box)
                split_lines.sort(key=lambda lb: lb[1])
                current_y = oy
                remaining_height = oh
                # There is lower tolerance since it is way less likely to have noise slightly out of box
                # Than have a random stray box
                # However upper_height and remaining_height can be adjusted if needed
                for line in split_lines:
                    ly = line[1]
                    lh_line = line[3]
                    upper_height = ly - current_y
                    if upper_height >= 15:
                        new_other_boxes.append((ox, current_y, ow, upper_height))
                    current_y = ly + lh_line
                    remaining_height = oh - (current_y - oy)
                if remaining_height >= 15:
                    new_other_boxes.append((ox, current_y, ow, remaining_height))

            remaining_line_boxes = [lb for lb in line_boxes if lb not in used_lines]
            final_boxes = new_other_boxes + remaining_line_boxes
            final_boxes = sorted(final_boxes, key=sort_key)  # Re-sort after splitting

            for idx, (x, y, w, h) in enumerate(final_boxes, start=1):
                #Writing cut boxes to files in sorted order for next steps.
                roi = imageLoad[y:y + h, x:x + w]
                color = (0, 0, 255) if is_horizontal_line((x, y, w, h)) else (36, 255, 12)
                cv2.imwrite(f"{input_dir}/roi{idx}.png", roi)
                cv2.rectangle(imageLoad, (x, y), (x + w, y + h), color, 2)

            if DEBUG_MODE:
                #We can check what we have drawn there
                cv2.imwrite(f"{input_dir}/boxed.png", imageLoad)

            #Processing even more for higher accuracy ocr, details in implementations of those functions
            process_images()
            preprocessWords()
        except Exception as e:
            print(f"Unexpected error: {e}")


class OCRDataset(Dataset):
    """Custom dataset for loading OCR images and labels"""

    def __init__(self, txt_file: str, img_root: str, tokenizer: Model, img_size=(128, 32)):
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

#Temporary
def train_model():
    """Training loop for OCR model"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize tokenizer and dataset
    tokenizer = Model()
    dataset = OCRDataset(
        txt_file=r"..\datasets\iam\iam_dataset\train_gt.txt",
        img_root=r"..\datasets\iam\iam_dataset",
        tokenizer=tokenizer
    )
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)

    # Model & optimizer
    model = OCRTransformer(vocab_size=tokenizer.vocab_size()).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Load existing checkpoint if exists
    checkpoint_path = r".\model.pth"
    if os.path.exists(checkpoint_path):
        print(f"Loading model from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint)
        # Optional: load optimizer state if previously saved
        # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # Loss function
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.char2idx[tokenizer.pad_token])

    # Training loop
    for epoch in range(100):
        model.train()
        total_loss = 0

        for images, targets, lengths in dataloader:
            #Mostly magic formulas from smarter people, no idea what can I change to make it better
            #Neither do I know how it works
            images, targets = images.to(device), targets.to(device)
            tgt_input = targets[:, :-1]
            tgt_output = targets[:, 1:]

            logits = model(images, tgt_input)
            logits = logits.reshape(-1, logits.shape[-1])
            tgt_output = tgt_output.reshape(-1)

            loss = loss_fn(logits, tgt_output)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(dataloader):.4f}")

        # Save model
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Model saved to {checkpoint_path}")
