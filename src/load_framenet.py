import nltk
import pandas as pd
import os
from tqdm import tqdm

def download_framenet():
    try:
        from nltk.corpus import framenet as fn
        fn.corpus_revision() # Just to trigger loading
    except Exception:
        print("Downloading FrameNet dataset...")
        nltk.download('framenet_v17')

def load_framenet_data(full=False, max_sentences=10000, top_k_verbs=100, samples_per_frame=30):
    """
    Extracts sentences, target predicates, frame labels, and frame elements.
    If full=False, restricts to top_k_verbs and max_sentences.
    """
    download_framenet()
    from nltk.corpus import framenet as fn
    
    print("Collecting verb targets...")
    # Find most frequent verbs
    verb_counts = []
    
    if not full:
        print(f"Finding top {top_k_verbs} verbs...")
        for lu in tqdm(fn.lus()):
            if lu.name.endswith('.v'):
                verb_counts.append((lu.name, len(lu.exemplars)))
        
        verb_counts.sort(key=lambda x: x[1], reverse=True)
        top_verbs = set([v[0] for v in verb_counts[:top_k_verbs]])
    else:
        top_verbs = None
        
    print("Extracting exemplars...")
    data = []
    sentence_count = 0
    reached_max = False
    
    for lu in tqdm(fn.lus()):
        if not lu.name.endswith('.v'):
            continue
            
        if not full and lu.name not in top_verbs:
            continue
            
        frame_name = lu.frame.name
        predicate = lu.name.split('.')[0]
        
        for exemplar in lu.exemplars:
            if not full and sentence_count >= max_sentences:
                reached_max = True
                break
                
            text = exemplar.text
            
            # Extract roles
            roles = []
            if hasattr(exemplar, 'FE'): # Frame Elements
                # It is usually a tuple of (start, end, role_name)
                # We'll extract the role text
                for fe in exemplar.FE[0] if isinstance(exemplar.FE, tuple) and len(exemplar.FE) > 0 and isinstance(exemplar.FE[0], tuple) else exemplar.FE:
                    if isinstance(fe, tuple) and len(fe) == 3:
                        start, end, role_name = fe
                        roles.append({"role": role_name, "text": text[start:end]})
                    elif hasattr(fe, 'items'):
                        pass # Depends on dict structure
            
            data.append({
                "sentence": text,
                "target_predicate": predicate,
                "frame_label": frame_name,
                "frame_elements": roles
            })
            
            sentence_count += 1
            
        if reached_max:
            break

    df = pd.DataFrame(data)
    
    if len(df) > 0 and samples_per_frame is not None:
        df = df.groupby('frame_label', group_keys=False).apply(
            lambda x: x.sample(n=min(len(x), samples_per_frame))
        ).reset_index(drop=True)

    return df

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Load FrameNet data")
    parser.add_argument("--full", action="store_true", help="Load full dataset instead of subset")
    parser.add_argument("--out", type=str, default="../data/framenet_subset.csv", help="Output path")
    args = parser.parse_args()
    
    print(f"Loading {'full' if args.full else 'subset'} FrameNet dataset...")
    df = load_framenet_data(full=args.full)
    print(f"Extracted {len(df)} sentences.")
    
    # Save the dataframe
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Saved to {args.out}")
