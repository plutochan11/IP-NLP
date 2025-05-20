import os
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import re
from nltk.translate import meteor_score
import nltk
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine
import datetime

# Install required NLTK data if not already present
try:
    nltk.data.find('punkt')
except LookupError:
    nltk.download('punkt')
    
# --- Configuration ---
MODEL_DIR = "Fine-tuned Results/tiny-stories-paraphrase-finetuned"
print(f"Using model from: {MODEL_DIR}")

# Test sentences for paraphrasing
test_sentences = [
    "The cat sat on the mat.",
    "She enjoys reading books in her free time.",
    "The weather is nice today, let's go for a walk.",
    "Mary and John are good friends, and they like to play toys together.",
    "The squirrel climbed the tall tree and looked for nuts.",
    "The toy car rolled down the hill very fast.",
    "The chef prepared a delicious meal for dinner.",
    "Children were playing in the park all afternoon.",
    "The teacher explained the complex math problem to the students.",
    "The old bridge connects two small villages across the river.",
    "My grandmother baked cookies with chocolate chips yesterday.",
    "The musician practiced the piano for three hours straight.",
    "Scientists discovered a new species in the Amazon rainforest.",
    "The detective solved the mystery by finding crucial evidence.",
    "Farmers wake up early to tend to their crops and animals.",
    "The astronaut looked down at Earth from the space station.",
    "Birds build nests in trees to protect their eggs and babies.",
    "The artist painted a beautiful landscape of the mountains.",
    "The football team celebrated after winning the championship.",
    "A strong wind blew through the valley, shaking the trees."
]

# Reference paraphrases for some test sentences (for automated evaluation)
reference_paraphrases = {
    "The cat sat on the mat.": ["A cat was sitting on the mat.", "The feline rested on the floor covering."],
    "She enjoys reading books in her free time.": ["In her leisure time, she likes to read books.", "Reading books is her favorite leisure activity."],
    "The weather is nice today, let's go for a walk.": ["It's a beautiful day, we should take a walk.", "The pleasant weather makes it perfect for walking outside."],
    "Mary and John are good friends, and they like to play toys together.": ["Mary and John are close friends who enjoy playing with toys together.", "John and Mary share a friendship and have fun playing with toys."],
    "The squirrel climbed the tall tree and looked for nuts.": ["The agile squirrel ascended the lofty tree in search of acorns.", "A squirrel went up a high tree to find some nuts."],
    "The toy car rolled down the hill very fast.": ["The miniature automobile zoomed down the slope with great rapidity.", "The small toy vehicle descended the hill at high speed."],
    "The chef prepared a delicious meal for dinner.": ["A tasty dinner was cooked by the chef.", "For the evening meal, the chef made something scrumptious."],
    "Children were playing in the park all afternoon.": ["Kids spent the whole afternoon having fun in the park.", "The park was filled with children at play throughout the afternoon."],
    "The teacher explained the complex math problem to the students.": ["The difficult mathematics question was clarified by the teacher for the class.", "Students received an explanation of the challenging math problem from their teacher."],
    "The old bridge connects two small villages across the river.": ["A pair of tiny hamlets are linked by an ancient bridge that spans the river.", "The aged crossing joins two minor settlements on opposite sides of the waterway."],
    "My grandmother baked cookies with chocolate chips yesterday.": ["Yesterday, my grandma made chocolate chip cookies.", "Chocolate-studded biscuits were prepared by my grandmother the previous day."],
    "The musician practiced the piano for three hours straight.": ["For three consecutive hours, the piano was rehearsed by the musician.", "The pianist spent three uninterrupted hours practicing their instrument."],
    "Scientists discovered a new species in the Amazon rainforest.": ["A previously unknown organism was found by researchers in the Amazonian jungle.", "In the rainforests of the Amazon, the scientific community identified a species not seen before."],
    "The detective solved the mystery by finding crucial evidence.": ["The key clue allowed the detective to resolve the case.", "By uncovering vital proof, the investigator cracked the mystery."],
    "Farmers wake up early to tend to their crops and animals.": ["Agricultural workers rise at dawn to care for their livestock and plantations.", "The people who farm get out of bed before sunrise to look after their plants and farm animals."],
    "The astronaut looked down at Earth from the space station.": ["From the orbital platform, the space traveler gazed at our planet below.", "The Earth was observed from above by the astronaut aboard the station in space."],
    "Birds build nests in trees to protect their eggs and babies.": ["Avian creatures construct homes in the branches to safeguard their offspring and unhatched young.", "To keep their chicks and eggs safe, birds create nests among tree limbs."],
    "The artist painted a beautiful landscape of the mountains.": ["A stunning mountain vista was captured on canvas by the painter.", "The mountains' beauty was depicted in an artwork created by the artist."],
    "The football team celebrated after winning the championship.": ["Following their championship victory, the soccer squad rejoiced.", "The team that plays football had a celebration after becoming champions."],
    "A strong wind blew through the valley, shaking the trees.": ["The valley's trees trembled as powerful gusts of air rushed through.", "Forceful breezes coursed through the valley, causing the trees to quiver."]
}

# --- Load Model and Tokenizer ---
print("Loading model and tokenizer...")
try:
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    print("Model and tokenizer loaded successfully.")
    
    # Get device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = model.to(device)
    print(f"Using device: {device}")
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

# --- Load Sentence Transformer for semantic similarity ---
print("Loading sentence transformer for semantic evaluation...")
try:
    semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
    semantic_model = semantic_model.to(device)
except Exception as e:
    print(f"Error loading sentence transformer: {e}")
    print("Will skip semantic similarity evaluation")
    semantic_model = None

# --- Format Input Using the Same Template as in Training ---
def format_instruction(input_text):
    """Format the input text exactly as in the fine-tuning script"""
    instruction = "Rewrite the provided text in different words while keeping the core meaning the same."
    
    # Add the "Paraphrase: " prefix as in the training data
    input_with_prefix = f"Paraphrase: {input_text}"
    
    return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input_with_prefix}

### Response:
"""

# --- Generate Paraphrases ---
print("\nGenerating paraphrases...")
results = []

for sentence in test_sentences:
    # Format using the same template as in training
    prompt = format_instruction(sentence)
    
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    # Generate with good parameters for paraphrasing
    outputs = model.generate(
        **inputs,
        max_length=150,
        num_beams=5,
        do_sample=True,
        temperature=0.7,
        early_stopping=True,
        no_repeat_ngram_size=2
    )
    
    raw_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract the paraphrase - everything after "### Response:"
    match = re.search(r"### Response:(.*?)(?:$|###)", raw_output, re.DOTALL)
    if match:
        paraphrase = match.group(1).strip()
    else:
        parts = raw_output.split("### Response:")
        if len(parts) > 1:
            paraphrase = parts[1].strip()
        else:
            paraphrase = raw_output.replace(prompt, "").strip()
    
    # Store the result
    result = {
        "original": sentence,
        "generated": paraphrase
    }
    
    # If reference paraphrase exists, add it
    if sentence in reference_paraphrases:
        result["references"] = reference_paraphrases[sentence]
    else:
        result["references"] = []
        
    results.append(result)
    
    # Just print a progress indicator
    print(f"Processing test case {len(results)}/{len(test_sentences)}...", end="\r")

# --- Evaluate Paraphrases ---
print("\nEvaluating paraphrases...")

# 1. Calculate automatic metrics when references are available
filtered_results = [r for r in results if r["references"]]
if filtered_results:
    # METEOR Score (sensitive to synonyms)
    meteor_scores = []
    for pred, refs in zip([r["generated"] for r in filtered_results], 
                         [r["references"] for r in filtered_results]):
        pred_tokens = nltk.word_tokenize(pred.lower())
        refs_tokens = [nltk.word_tokenize(ref.lower()) for ref in refs]
        score = meteor_score.meteor_score(refs_tokens, pred_tokens)
        meteor_scores.append(score)

# 2. Semantic similarity using sentence embeddings
if semantic_model:
    semantic_scores = []
    
    for result in results:
        original_embedding = semantic_model.encode(result["original"])
        generated_embedding = semantic_model.encode(result["generated"])
        
        # Calculate cosine similarity (1 - cosine distance)
        similarity = 1 - cosine(original_embedding, generated_embedding)
        semantic_scores.append(similarity)

# 3. Lexical diversity (how different are the words used)
diversity_scores = []

for result in results:
    original_tokens = set(nltk.word_tokenize(result["original"].lower()))
    generated_tokens = set(nltk.word_tokenize(result["generated"].lower()))
    
    # Common tokens
    common = original_tokens.intersection(generated_tokens)
    
    # Jaccard distance (higher means more different vocabulary)
    if len(original_tokens.union(generated_tokens)) > 0:
        jaccard_distance = 1 - len(common) / len(original_tokens.union(generated_tokens))
        diversity_scores.append(jaccard_distance)

# 4. Length ratio (paraphrase shouldn't be too much longer/shorter)
length_ratios = []

for result in results:
    original_length = len(result["original"].split())
    generated_length = len(result["generated"].split())
    
    ratio = generated_length / original_length if original_length > 0 else 0
    length_ratios.append(ratio)

# 5. Overall summary
print("\n--- OVERALL EVALUATION SUMMARY ---")
print(f"Number of test sentences: {len(results)}")

# Prepare summary metrics
summary_metrics = {
    "num_test_sentences": len(results),
    "meteor_score": np.mean(meteor_scores) if filtered_results else None,
    "semantic_similarity": np.mean(semantic_scores) if semantic_model else None,
    "lexical_diversity": np.mean(diversity_scores),
    "length_ratio": np.mean(length_ratios)
}

if filtered_results:
    print(f"METEOR Score: {summary_metrics['meteor_score']:.4f}")

if semantic_model:
    print(f"Semantic Similarity: {summary_metrics['semantic_similarity']:.4f} (higher is better, ideal > 0.7)")

print(f"Lexical Diversity: {summary_metrics['lexical_diversity']:.4f} (higher means more word changes, ideal 0.6-0.8)")
print(f"Length Ratio: {summary_metrics['length_ratio']:.2f} (ideal: 0.8-1.2)")

# Save results to file
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "Eval. Results")
results_file = os.path.join(results_dir, f"evaluation_results_{timestamp}.txt")

with open(results_file, "w") as f:
    # Write header
    f.write(f"PARAPHRASE EVALUATION RESULTS - {timestamp}\n")
    f.write(f"Model: {MODEL_DIR}\n")
    f.write("=" * 80 + "\n\n")
    
    # Write sentence comparisons
    f.write("SENTENCE COMPARISONS\n")
    f.write("-" * 80 + "\n\n")
    
    for i, result in enumerate(results, 1):
        f.write(f"Example {i}:\n")
        f.write(f"Original: {result['original']}\n")
        f.write(f"Generated: {result['generated']}\n")
        if result['references']:
            f.write(f"Reference: {result['references'][0]}\n")
        f.write("\n")
    
    # Write summary metrics
    f.write("\nSUMMARY METRICS\n")
    f.write("-" * 80 + "\n\n")
    f.write(f"Number of test sentences: {summary_metrics['num_test_sentences']}\n")
    
    if summary_metrics['meteor_score'] is not None:
        f.write(f"METEOR Score: {summary_metrics['meteor_score']:.4f}\n")
    
    if summary_metrics['semantic_similarity'] is not None:
        f.write(f"Semantic Similarity: {summary_metrics['semantic_similarity']:.4f} (higher is better, ideal > 0.7)\n")
    
    f.write(f"Lexical Diversity: {summary_metrics['lexical_diversity']:.4f} (higher means more word changes, ideal 0.6-0.8)\n")
    f.write(f"Length Ratio: {summary_metrics['length_ratio']:.2f} (ideal: 0.8-1.2)\n\n")
    
    f.write("Note: These automated metrics provide a general assessment, but human evaluation\n")
    f.write("of paraphrasing quality is still the gold standard.\n")

print(f"\nDetailed results and comparisons saved to: {results_file}")