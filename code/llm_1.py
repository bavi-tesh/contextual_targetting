import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
import pdfplumber
from bs4 import BeautifulSoup
import requests
import re
from PIL import Image
import pytesseract
from pytesseract import TesseractNotFoundError
from tqdm import tqdm
import cv2

# Function to check Tesseract installation
def check_tesseract():
    try:
        pytesseract.get_tesseract_version()
        print("Tesseract is installed.")
    except TesseractNotFoundError:
        raise EnvironmentError("Tesseract is not installed or it's not in your PATH. Please install Tesseract.")

# Function to process text input
def process_text_input(text):
    return text

# Function to process text from a file
def process_text_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    return text

# Function to process image and extract text
def process_image_to_text(image_path):
    image = Image.open(image_path)
    text = pytesseract.image_to_string(image)
    return text

# Function to process PDF and extract text
def process_pdf_to_text(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        pdf_text = ""
        for page in pdf.pages:
            pdf_text += page.extract_text()
    return pdf_text

# Function to process URL and extract text
def process_url_to_text(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")
    paragraphs = soup.find_all("p")
    url_text = " ".join([para.get_text() for para in paragraphs])
    url_text = re.sub(r'\s+', ' ', url_text)  # Remove extra whitespace
    return url_text

# Function to process video and extract text from frames
def process_video_to_text(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_texts = []
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    for _ in tqdm(range(frame_count), desc="Processing video frames"):
        ret, frame = cap.read()
        if not ret:
            break
        image = Image.fromarray(frame)
        text = pytesseract.image_to_string(image)
        frame_texts.append(text)
    
    cap.release()
    video_text = " ".join(frame_texts)
    video_text = re.sub(r'\s+', ' ', video_text)  # Remove extra whitespace
    return video_text

# Function to preprocess files based on their type
def preprocess_files(input_type, input_paths):
    preprocessed_texts = []
    for input_path in input_paths:
        print(f"Processing {input_type} file: {input_path}")
        if input_type == "text":
            if os.path.isfile(input_path):
                preprocessed_texts.append(process_text_from_file(input_path))
            else:
                preprocessed_texts.append(process_text_input(input_path))
        elif input_type == "image":
            preprocessed_texts.append(process_image_to_text(input_path))
        elif input_type == "pdf":
            preprocessed_texts.append(process_pdf_to_text(input_path))
        elif input_type == "url":
            preprocessed_texts.append(process_url_to_text(input_path))
        elif input_type == "video":
            preprocessed_texts.append(process_video_to_text(input_path))
        else:
            raise ValueError(f"Unsupported input type: {input_type}")
    return preprocessed_texts

# Ensure Tesseract is installed
check_tesseract()

# Example usage with multiple files of different types
train_files = {
    "text": [
        "Adtech innovations like programmatic advertising have revolutionized how brands target consumers online. By leveraging data analytics and real-time bidding, advertisers can precisely reach their target audiences across multiple digital channels. This efficiency has significantly boosted ROI for many businesses, making ad spending more accountable and results-driven.",
        "As adtech continues to evolve, the debate around data sovereignty and consumer rights intensifies. Companies are grappling with the ethical implications of data collection and targeted advertising, especially in light of recent privacy scandals. Regulators are under pressure to enact stricter laws to protect user data, while advertisers seek innovative ways to maintain relevance without compromising privacy. The paragraph discusses the ongoing debate and challenges in adtech.",
        "Blockchain technology is being explored as a means to enhance transparency and trust in digital advertising. Proponents believe it can help combat ad fraud and improve accountability. However, the complexity and cost of implementing blockchain solutions are significant barriers to widespread adoption.",
        "Amidst the dimly lit alley, smoke curled lazily from the discarded cigarette, casting a haze over the scene. Two figures squared off, their movements sharp and deliberate in the glow of flickering street lamps. The air crackled with tension as fists clenched and insults flew, each word escalating the conflict. It was a raw display of anger and pride, a moment where adrenaline surged and consequences blurred.",
        "Under the canopy of stars, they stood hand in hand, the night alive with whispered secrets and the gentle rustle of leaves. Moonlight painted patterns on the path before them, a silent witness to their quiet exchange. In that fleeting moment, words were unnecessary; their hearts spoke in the language of shared dreams and unspoken promises. It was a scene straight from a romance novel, where time stood still in the embrace of love."
    ],
    "image": [
        r"C:\Users\mbavi\Downloads\AD3X\Images\Image1.jpg",
        r"C:\Users\mbavi\Downloads\AD3X\Images\Image2.jpg",
        r"C:\Users\mbavi\Downloads\AD3X\Images\Image3.jpg",
        r"C:\Users\mbavi\Downloads\AD3X\Images\Image4.jpg",
        r"C:\Users\mbavi\Downloads\AD3X\Images\Image5.jpg"
        
    ],
    "pdf": [
        r"C:\Users\mbavi\Downloads\AD3X\PDFs\PDF_1.pdf",
        r"C:\Users\mbavi\Downloads\AD3X\PDFs\PDF_2.pdf",
        r"C:\Users\mbavi\Downloads\AD3X\PDFs\PDF_3.pdf",
        r"C:\Users\mbavi\Downloads\AD3X\PDFs\PDF_4.pdf",
        r"C:\Users\mbavi\Downloads\AD3X\PDFs\PDF_5.pdf"
    ],
    'url': [
        r"https://www.alkimi.org/privacy-policy",
        r"https://www.wikipedia.org/",
        r"https://www.openai.com/",
        r"https://www.github.com/",
        r"https://www.stackoverflow.com/"
    ],
    'video': [
        r"C:\Users\mbavi\Downloads\AD3X\Videos\Video1.mp4",
        r"C:\Users\mbavi\Downloads\AD3X\Videos\Video2.mp4",
        r"C:\Users\mbavi\Downloads\AD3X\Videos\Video3.mp4",
        r"C:\Users\mbavi\Downloads\AD3X\Videos\Video4.mp4",
        r"C:\Users\mbavi\Downloads\AD3X\Videos\Video5.mp4"
    ]
}

validation_files = {
    "text": [
        "Additional text for validation."
    ],
    "image": [
        r"C:\Users\mbavi\Downloads\AD3X\Images\Image6.jpg",
        r"C:\Users\mbavi\Downloads\AD3X\Images\Image7.jpg"
    ],
    "pdf": [
        r"C:\Users\mbavi\Downloads\AD3X\PDFs\PDF_6.pdf"
    ],
    'url': [
        r"https://www.w3schools.com/",
        r"https://www.wix.com"
    ],
    'video': [
        r"C:\Users\mbavi\Downloads\AD3X\Videos\Video6.mp4"
    ]
}

# Combine all preprocessed texts into a single dataset
def combine_dataset(input_files):
    combined_dataset = []
    for input_type, paths in input_files.items():
        preprocessed_texts = preprocess_files(input_type, paths)
        for text in preprocessed_texts:
            combined_dataset.append((text, input_type))
    return combined_dataset

train_dataset = combine_dataset(train_files)
val_dataset = combine_dataset(validation_files)

# Initialize tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Tokenize texts
train_tokenized_texts = tokenizer([text[0] for text in train_dataset], padding=True, truncation=True, return_tensors="pt")
val_tokenized_texts = tokenizer([text[0] for text in val_dataset], padding=True, truncation=True, return_tensors="pt")

# Prepare labels (for example purposes, assign numeric labels to source types)
source_to_label = {"text": 0, "image": 1, "pdf": 2, "url": 3, "video": 4}
train_labels = [source_to_label[text[1]] for text in train_dataset]
val_labels = [source_to_label[text[1]] for text in val_dataset]

# Create custom dataset
class CustomDataset(Dataset):
    def __init__(self, tokenized_texts, labels):
        self.input_ids = tokenized_texts["input_ids"]
        self.attention_masks = tokenized_texts["attention_mask"]
        self.labels = torch.tensor(labels)
        
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_masks[idx],
            "labels": self.labels[idx]
        }

# Create datasets and DataLoaders
train_dataset = CustomDataset(train_tokenized_texts, train_labels)
val_dataset = CustomDataset(val_tokenized_texts, val_labels)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

# Initialize model
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=len(source_to_label))

# Define optimizer
optimizer = AdamW(model.parameters(), lr=1e-5)

# Training loop with validation
epochs = 3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

for epoch in range(epochs):
    # Training
    model.train()
    for batch in train_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
    
    # Validation
    model.eval()
    val_losses = []
    val_predictions = []
    val_targets = []
    
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            val_losses.append(outputs.loss.item())
            val_predictions.extend(torch.argmax(outputs.logits, dim=1).tolist())
            val_targets.extend(labels.tolist())
    
    avg_val_loss = sum(val_losses) / len(val_losses)
    val_accuracy = sum(1 for x, y in zip(val_predictions, val_targets) if x == y) / len(val_targets)
    
    print(f"Epoch {epoch + 1}/{epochs}")
    print(f"Training Loss: {loss.item():.4f}")
    print(f"Validation Loss: {avg_val_loss:.4f} | Validation Accuracy: {val_accuracy:.2%}")
    print("=" * 50)

print("Training finished.")
