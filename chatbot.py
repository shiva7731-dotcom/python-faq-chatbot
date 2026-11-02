import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import string

# Download required NLTK resources (stopwords, punkt tokenizer)
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Structured FAQ Database
faq_database = {
    "ecommerce": [
        {
            "question": "How can I track my order status?",
            "answer": "You can track your order in real-time by going to 'My Account' -> 'Orders' and clicking on the tracking link provided next to your shipment."
        },
        {
            "question": "What is your return and refund policy?",
            "answer": "We offer a 30-day hassle-free return policy on all unused items in their original packaging. Refunds are processed within 3-5 business days after inspection."
        },
        {
            "question": "What payment methods do you accept?",
            "answer": "We accept all major credit/debit cards (Visa, MasterCard, Amex), PayPal, Apple Pay, Google Pay, and Cash on Delivery (COD) for eligible locations."
        },
        {
            "question": "How long does standard shipping take?",
            "answer": "Standard shipping typically takes 3 to 5 business days within the continental region. Expedited 2-day shipping options are also available at checkout."
        },
        {
            "question": "Can I cancel or modify my order after placing it?",
            "answer": "Orders can be canceled or modified within 1 hour of placement through your order dashboard before they enter our fulfillment queue."
        }
    ]
}

class FAQChatbot:
    def __init__(self, faqs):
        self.faqs = faqs
        self.questions = [faq["question"] for faq in self.faqs]
        self.answers = [faq["answer"] for faq in self.faqs]
        
        # Initialize TF-IDF Vectorizer
        self.vectorizer = TfidfVectorizer(tokenizer=self.preprocess, token_pattern=None)
        # Fit vectorizer on FAQ questions
        self.tfidf_matrix = self.vectorizer.fit_transform(self.questions)

    def preprocess(self, text):
        """Preprocess text: lowercase, remove punctuation, tokenize, and remove stopwords."""
        text = text.lower()
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        # Tokenize
        tokens = word_tokenize(text)
        # Remove stopwords
        stop_words = set(stopwords.words('english'))
        filtered_tokens = [word for word in tokens if word not in stop_words]
        return filtered_tokens

    def get_response(self, user_query, threshold=0.15):
        """Match user query with FAQs using Cosine Similarity."""
        # Transform user query into TF-IDF vector
        query_vector = self.vectorizer.transform([user_query])
        
        # Calculate cosine similarity between query and all FAQ questions
        similarity_scores = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
        
        # Find best match index
        best_match_idx = similarity_scores.argmax()
        best_score = similarity_scores[best_match_idx]

        if best_score >= threshold:
            return self.answers[best_match_idx], self.questions[best_match_idx], best_score
        else:
            return "I'm sorry, I couldn't find a direct answer to your question in our FAQ database. Please try rephrasing.", None, best_score

# CLI Chat Interface Loop
if __name__ == "__main__":
    print("=" * 60)
    print("🤖 Welcome to the Python FAQ Chatbot (Powered by NLTK & Scikit-Learn)")
    print("=" * 60)
    print("Type 'exit' or 'quit' to end the conversation.
")
    
    bot = FAQChatbot(faq_database["ecommerce"])
    
    while True:
        user_input = input("You: ")
        if user_input.lower() in ['exit', 'quit']:
            print("Chatbot: Goodbye! Have a great day!")
            break
            
        if not user_input.strip():
            continue
            
        answer, matched_q, score = bot.get_response(user_input)
        
        print(f"Bot: {answer}")
        if matched_q:
            print(f"     [Matched FAQ: '{matched_q}' | Similarity Score: {score:.2f}]\n")
        else:
            print()
