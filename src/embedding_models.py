import torch
from transformers import BertTokenizer, BertModel
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
import spacy
import numpy as np

class EmbeddingGenerator:
    def __init__(self, device='cpu'):
        self.device = device
        
        # Load spaCy for Word2Vec-style vectors
        print("Loading spacy model for vectors...")
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except:
            print("Warning: en_core_web_sm not found for spaCy vectors. It must be downloaded.")
            self.nlp = None
            
        # Load sentence-transformers MiniLM
        print("Loading sentence-transformers/all-MiniLM-L6-v2...")
        self.minilm = SentenceTransformer('all-MiniLM-L6-v2', device=self.device)
        
        # Load HuggingFace BERT
        print("Loading bert-base-uncased...")
        self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert_model = BertModel.from_pretrained('bert-base-uncased').to(self.device)
        self.bert_model.eval()
        
        # TF-IDF
        self.tfidf_vectorizer = TfidfVectorizer(max_features=5000)
        self.tfidf_fitted = False

    def generate_tfidf(self, sentences):
        if not self.tfidf_fitted:
            print("Fitting TF-IDF...")
            self.tfidf_vectorizer.fit(sentences)
            self.tfidf_fitted = True
            
        embeddings = self.tfidf_vectorizer.transform(sentences).toarray()
        return embeddings
        
    def generate_spacy_vectors(self, sentences):
        embeddings = []
        for doc in self.nlp.pipe(sentences):
            embeddings.append(doc.vector)
        return np.array(embeddings)
        
    def generate_minilm(self, sentences, triggers=None, batch_size=32):
        print("Generating MiniLM embeddings...")
        if triggers is not None:
            if len(sentences) != len(triggers):
                raise ValueError("sentences and triggers must have the same length")
            inputs = [f"{s} [SEP] {t}" for s, t in zip(sentences, triggers)]
        else:
            inputs = sentences
        embeddings = self.minilm.encode(inputs, batch_size=batch_size, convert_to_numpy=True)
        return embeddings
        
    def generate_bert(self, sentences, batch_size=32):
        print("Generating BERT embeddings...")
        embeddings = []
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i+batch_size]
            encoded = self.bert_tokenizer(batch, padding=True, truncation=True, return_tensors='pt', max_length=128)
            encoded = {k: v.to(self.device) for k, v in encoded.items()}
            
            with torch.no_grad():
                outputs = self.bert_model(**encoded)
                # Mean pooling
                attention_mask = encoded['attention_mask']
                token_embeddings = outputs.last_hidden_state
                input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
                sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                pooled = sum_embeddings / sum_mask
                
                embeddings.append(pooled.cpu().numpy())
                
        return np.vstack(embeddings)

    def extract_predicate_window(self, sentence, target_predicate, window_size=2):
        """ Extract a textual window around the target predicate for context-aware embedding """
        tokens = sentence.split()
        target_idx = -1
        for i, t in enumerate(tokens):
            if t.lower().startswith(target_predicate.lower()):
                target_idx = i
                break
                
        if target_idx == -1:
            return sentence # Fallback
            
        start = max(0, target_idx - window_size)
        end = min(len(tokens), target_idx + window_size + 1)
        return " ".join(tokens[start:end])

    def get_embeddings(self, model_type, data, context_type="sentence"):
        """
        model_type: 'tfidf', 'spacy', 'minilm', 'bert'
        context_type: 'sentence', 'window'
        """
        # data is expected to be a dataframe with 'sentence' and 'target_predicate' columns
        if context_type == "window":
            sentences = [self.extract_predicate_window(row['sentence'], row['target_predicate']) 
                         for _, row in data.iterrows()]
        else:
            sentences = data['sentence'].tolist()
            
        if model_type == 'tfidf':
            return self.generate_tfidf(sentences)
        elif model_type == 'spacy':
            return self.generate_spacy_vectors(sentences)
        elif model_type == 'minilm':
            triggers = data['target_predicate'].tolist()
            return self.generate_minilm(sentences, triggers=triggers)
        elif model_type == 'bert':
            return self.generate_bert(sentences)
        else:
            raise ValueError(f"Unknown model_type {model_type}")
