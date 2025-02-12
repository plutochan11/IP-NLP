from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

model_name = 'roneneldan/TinyStories-33M'
tokenizer_name = "EleutherAI/gpt-neo-125M"

# Load the model and tokenizer
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

input_text = "Mary and John are good friends, and they like to play toys together."
# input_ids = tokenizer.encode(input_text, return_tensors="pt")
# output = model.generate(input_ids, max_length = 35)
# output_text = tokenizer.decode(output[0], skip_special_tokens=True)

# print (f"Original: {input_text}")
# print (f"Paraphrase: {output_text}")

# Paraphrase using pipeline
paraphrase_pipeline = pipeline('text-generation', model = model, tokenizer=tokenizer)
# para_pl = pipeline('text2text-generation', model = model, tokenizer=tokenizer)

outputs = paraphrase_pipeline (input_text, max_length = 35, num_return_sequences = 5, 
                               do_sample = True, top_k = 25, top_p = 0.88, truncation = True)

# Sampling
# outputs = paraphrase_pipeline (input_text, max_length = 35, num_return_sequences = 5, 
#                                do_sample = True, top_k = 25, top_p = 0.88, truncation = True)

# Output the results
for i, output in enumerate (outputs):
    print (f"Paraphrase {i + 1} : {output['generated_text']}")