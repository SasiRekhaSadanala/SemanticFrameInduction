import spacy
import networkx as nx

class FeatureExtractor:
    def __init__(self, model_name="en_core_web_sm"):
        """
        Loads the spaCy model.
        Run `python -m spacy download en_core_web_sm` beforehand.
        """
        print(f"Loading spaCy model: {model_name}")
        self.nlp = spacy.load(model_name)
    
    def process_sentence(self, sentence, target_predicate):
        """
        Parses the sentence and extracts the predicate and its syntactic arguments.
        Returns the spaCy doc and a dictionary of extracted arguments.
        """
        doc = self.nlp(sentence)
        target_predicate = target_predicate.lower().strip()
        
        predicate_token = None
        for token in doc:
            if token.lemma_.lower() == target_predicate or token.text.lower() == target_predicate:
                predicate_token = token
                break
        
        arguments = {}
        if predicate_token:
            for child in predicate_token.children:
                # Common dependency labels for arguments
                if child.dep_ in ['nsubj', 'nsubjpass', 'csubj', 'csubjpass']:
                    arguments['subject'] = child.text
                elif child.dep_ in ['dobj', 'pobj', 'iobj']:
                    arguments['object'] = child.text
                elif child.dep_ == 'prep':
                    # Extract the prepositional object attached to the preposition
                    for grandchild in child.children:
                        if grandchild.dep_ == 'pobj':
                            arguments[f'prep_{child.text}'] = grandchild.text
                elif child.dep_ in ['advmod', 'amod']:
                    arguments['modifier'] = child.text
                    
        return doc, predicate_token, arguments

    def build_predicate_argument_graph(self, documents, target_predicates):
        """
        Builds a NetworkX graph across multiple sentences.
        Nodes are predicates and arguments.
        Edges are syntactic relations.
        """
        G = nx.Graph()
        
        for doc_text, target in zip(documents, target_predicates):
            doc, pred_token, args = self.process_sentence(doc_text, target)
            pred_node = f"PRED_{target}"
            G.add_node(pred_node, type='predicate')
            
            for rel, arg_text in args.items():
                arg_node = f"ARG_{arg_text.lower()}"
                G.add_node(arg_node, type='argument')
                # Increment edge weight if exists
                if G.has_edge(pred_node, arg_node):
                    G[pred_node][arg_node]['weight'] += 1
                else:
                    G.add_edge(pred_node, arg_node, weight=1, relation=rel)
                    
        return G

if __name__ == "__main__":
    extractor = FeatureExtractor()
    sentence = "John bought a car from Mary."
    doc, pred, args = extractor.process_sentence(sentence, "buy")
    print(f"Predicate: buy")
    print(f"Arguments: {args}")
