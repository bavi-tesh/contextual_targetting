#bert-base-uncased (via transformers): BERT model for sentiment analysis.
#distilbert-base-uncased (via transformers): DistilBERT model for text classification.
# os: Operating system functionalities.
#torch: PyTorch library for machine learning tasks.
#PIL: Python Imaging Library for image processing.
#requests: HTTP library for making requests.
# BeautifulSoup: Library for parsing HTML and XML documents.
#fitz: PyMuPDF for PDF processing.
#spacy: Natural language processing library for tokenization and lemmatization.
#cv2: OpenCV library for computer vision tasks.
#re: Regular expressions for text cleaning.
#nltk: Natural Language Toolkit for text processing (tokenization, sentiment analysis).
#transformers: Hugging Face library for using transformer models (BERT, DistilBERT).
#torchvision.transforms: PyTorch library for image transformations.
#concurrent.futures: Library for concurrent programming.
#bs4: Beautiful Soup for parsing HTML and XML documents.
#TextBlob: Library for processing textual data.
import os
import torch
from PIL import Image
import requests
from bs4 import BeautifulSoup
import fitz  # PyMuPDF for PDF processing
import spacy
import cv2
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from torchvision import transforms
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from nrclex import NRCLex
from concurrent.futures import ThreadPoolExecutor, as_completed
from fer import FER

# Ensure NLTK resources are downloaded
nltk.download('punkt')
nltk.download('stopwords')

# Load SpaCy English model
nlp = spacy.load('en_core_web_sm')

# Initialize sentiment analyzer
vader_analyzer = SentimentIntensityAnalyzer()

# Initialize emotion detection model
emotion_detector = FER(mtcnn=True)

# Initialize necessary transformers and other tools
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Define helper functions
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text.strip()

def tokenize_text(text):
    return word_tokenize(text)

def preprocess_text_with_spacy(text):
    doc = nlp(text.lower())
    tokens = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]
    return ' '.join(tokens)

def extract_text_from_url(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    paragraphs = soup.find_all('p')
    text = ' '.join([para.get_text() for para in paragraphs])
    return clean_text(text)

def extract_text_from_pdf(pdf_path):
    document = fitz.open(pdf_path)
    text = "".join([page.get_text() for page in document])
    return clean_text(text)

def preprocess_image(image_path):
    image = Image.open(image_path).convert('RGB')
    return transform(image)

def analyze_image_emotion(image_path):
    image = cv2.imread(image_path)
    emotion, score = emotion_detector.top_emotion(image)
    return {'emotion': emotion, 'score': score}

def preprocess_videos(video_paths, output_folder, seconds_per_frame=2):
    def create_output_folder(output_folder):
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

    def get_video_name(video_path):
        return os.path.splitext(os.path.basename(video_path))[0]

    def save_frame(frame, frame_filename):
        cv2.imwrite(frame_filename, frame)

    def extract_frames_from_video(video_path, output_folder, seconds_per_frame):
        cap = cv2.VideoCapture(video_path)
        count = 0
        frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
        video_name = get_video_name(video_path)
        success = True

        while success:
            success, frame = cap.read()
            if success and int(cap.get(cv2.CAP_PROP_POS_FRAMES)) % (frame_rate * seconds_per_frame) == 0:
                frame_filename = os.path.join(output_folder, f"{video_name}_frame{count}.jpg")
                save_frame(frame, frame_filename)
                count += 1
        cap.release()

    create_output_folder(output_folder)
    for video_path in video_paths:
        extract_frames_from_video(video_path, output_folder, seconds_per_frame)

# Analysis functions
def analyze_sentiment(text):
    blob = TextBlob(text)
    vader_scores = vader_analyzer.polarity_scores(text)
    return {
        'polarity': blob.sentiment.polarity,
        'subjectivity': blob.sentiment.subjectivity,
        'vader': vader_scores
    }

def analyze_emotion(text):
    emotion = NRCLex(text)
    return emotion.raw_emotion_scores

def analyze_age_appropriateness(text):
    inappropriate_words = {'violence', 'drugs', 'explicit', 'fuck', 'shit', 'damn','bitch'}
    words = tokenize_text(text)
    is_appropriate = not any(word in inappropriate_words for word in words)
    return 'Appropriate' if is_appropriate else 'Inappropriate'

def analyze_iab_category(text):
    # Placeholder implementation
    categories = {
        'adtech': 'Technology',
        'privacy': 'Law, Government, and Politics',
        'blockchain': 'Technology',
        'romance': 'Arts and Entertainment'
    }
    for keyword, category in categories.items():
        if keyword in text:
            return category
    return 'Unknown'

# Main function for processing all inputs
def process_all_inputs(image_paths, video_paths, urls, pdf_paths, texts):
    results = {}

    # Preprocess images
    image_results = []
    for image_path in image_paths:
        preprocessed_image = preprocess_image(image_path)
        image_emotion = analyze_image_emotion(image_path)
        image_results.append({
            'image_path': image_path,
            'preprocessed_shape': preprocessed_image.shape,
            'preprocessed_sample': preprocessed_image[0, 0, :5].tolist(),  # Sample pixel values for illustration
            'emotion': image_emotion
        })
    results['images'] = image_results

    # Preprocess videos
    output_folder = 'frames/'
    preprocess_videos(video_paths, output_folder)
    results['videos'] = output_folder

    # Process and analyze text from URLs
    url_results = []
    with ThreadPoolExecutor() as executor:
        url_tasks = [executor.submit(extract_text_from_url, url) for url in urls]
        url_texts = [task.result() for task in as_completed(url_tasks)]
    for url, text in zip(urls, url_texts):
        preprocessed_text = preprocess_text_with_spacy(text)
        sentiment = analyze_sentiment(preprocessed_text)
        emotion = analyze_emotion(preprocessed_text)
        age_appropriateness = analyze_age_appropriateness(preprocessed_text)
        iab_category = analyze_iab_category(preprocessed_text)
        url_results.append({
            'url': url,
            'text': text,
            'preprocessed_text': preprocessed_text,
            'sentiment': sentiment,
            'emotion': emotion,
            'age_appropriateness': age_appropriateness,
            'iab_category': iab_category
        })
    results['url_texts'] = url_results

    # Process and analyze text from PDFs
    pdf_results = []
    for pdf_path in pdf_paths:
        pdf_text = extract_text_from_pdf(pdf_path)
        preprocessed_text = preprocess_text_with_spacy(pdf_text)
        sentiment = analyze_sentiment(preprocessed_text)
        emotion = analyze_emotion(preprocessed_text)
        age_appropriateness = analyze_age_appropriateness(preprocessed_text)
        iab_category = analyze_iab_category(preprocessed_text)
        pdf_results.append({
            'pdf_path': pdf_path,
            'text': pdf_text,
            'preprocessed_text': preprocessed_text,
            'sentiment': sentiment,
            'emotion': emotion,
            'age_appropriateness': age_appropriateness,
            'iab_category': iab_category
        })
    results['pdf_texts'] = pdf_results

    # Process and analyze given texts
    text_results = []
    for text in texts:
        preprocessed_text = preprocess_text_with_spacy(text)
        sentiment = analyze_sentiment(preprocessed_text)
        emotion = analyze_emotion(preprocessed_text)
        age_appropriateness = analyze_age_appropriateness(preprocessed_text)
        iab_category = analyze_iab_category(preprocessed_text)
        text_results.append({
            'text': text,
            'preprocessed_text': preprocessed_text,
            'sentiment': sentiment,
            'emotion': emotion,
            'age_appropriateness': age_appropriateness,
            'iab_category': iab_category
        })
    results['texts'] = text_results

    return results

# Example usage
if __name__ == "__main__":
    image_paths = [
        r"C:\Users\mbavi\Downloads\AD3X\Images\image1.jpg",
        r"C:\Users\mbavi\Downloads\AD3X\Images\image2.jpg",
        r"C:\Users\mbavi\Downloads\AD3X\Images\image3.jpg",
        r"C:\Users\mbavi\Downloads\AD3X\Images\image4.jpg",
        r"C:\Users\mbavi\Downloads\AD3X\Images\image5.jpg"
    ]
    video_paths = [
        r"C:\Users\mbavi\Downloads\AD3X\Videos\video1.mp4",
        r"C:\Users\mbavi\Downloads\AD3X\Videos\video2.mp4",
        r"C:\Users\mbavi\Downloads\AD3X\Videos\video3.mp4"
    ]
    urls = [
        "https://www.alkimi.org/privacy-policy",
        "https://www.wikipedia.org/",
        "https://www.openai.com/",
        "https://www.github.com/",
        "https://www.stackoverflow.com/"
    ]
    pdf_paths = [
        r"C:\Users\mbavi\Downloads\AD3X\PDFs\PDF_1.pdf",
        r"C:\Users\mbavi\Downloads\AD3X\PDFs\PDF_2.pdf",
        r"C:\Users\mbavi\Downloads\AD3X\PDFs\PDF_3.pdf",
        r"C:\Users\mbavi\Downloads\AD3X\PDFs\PDF_4.pdf",
        r"C:\Users\mbavi\Downloads\AD3X\PDFs\PDF_5.pdf"
    ]
    texts = [
        "Adtech innovations like programmatic advertising have revolutionized how brands target consumers online. By leveraging data analytics and real-time bidding, advertisers can precisely reach their target audiences across multiple digital channels. This efficiency has significantly boosted ROI for many businesses, making ad spending more accountable and results-driven.",
        "As adtech continues to evolve, the debate around data sovereignty and consumer rights intensifies. Companies are grappling with the ethical implications of data collection and targeted advertising, especially in light of recent privacy scandals. Regulators are under pressure to enact stricter laws to protect user data, while advertisers seek innovative ways to maintain relevance without compromising privacyNeutral. The paragraph discusses the ongoing debate and challenges in adtech.",
        "Blockchain technology is being explored as a means to enhance transparency and trust in digital advertising. Proponents believe it can help combat ad fraud and improve accountability. However, the complexity and cost of implementing blockchain solutions are significant barriers to widespread adoption.",
        "Amidst the dimly lit alley, smoke curled lazily from the arded cigarette, casting a haze over the scene. Two figures squared off, their movements sharp and deliberate in the glow of flickering street lamps. The air crackled with tension as fists clenched and insults flew, each word escalating the conflict. It was a raw display of anger and pride, a moment where adrenaline surged and consequences blurred.",
        "Under the canopy of stars, they stood hand in hand, the night alive with whispered secrets and the gentle rustle of leaves. Moonlight painted patterns on the path before them, a silent witness to their quiet exchange. In that fleeting moment, words were unnecessary; their hearts spoke in the language of shared dreams and unspoken promises. It was a scene straight from a romance novel, where time stood still in the embrace of love.""Under the canopy of stars, they stood hand in hand, the night alive with whispered secrets and the gentle rustle of leaves. Moonlight painted patterns on the path before them, a silent witness to their quiet exchange. In that fleeting moment, words were unnecessary; their hearts spoke in the language of shared dreams and unspoken promises. It was a scene straight from a romance novel, where time stood still in the embrace of love."
    ]
    results = process_all_inputs(image_paths, video_paths, urls, pdf_paths, texts)
    print(results)
    