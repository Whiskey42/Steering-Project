from Embeddings import set_model_and_tokenizer, mean_pooling
from Steering_vector import import_feature_texts

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import logging
logging.set_verbosity_error()


def initialize_model_and_tokenizer(model_name):
    """
    Initialize the model and tokenizer for text generation.
    
    Args:
        model_name (str): Name of the pre-trained causal language model.
    
    Returns:
        tuple: (model, tokenizer) - The loaded model and tokenizer for text generation.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, output_hidden_states=True)
    return model, tokenizer


def display_next_tokens(model, tokenizer, prompt, k=10):
    """
    Display the next token predictions for a given prompt using the specified model.
    
    Args:
        model: The causal language model for text generation.
        tokenizer: The tokenizer corresponding to the model.
        prompt (str): Input text prompt for prediction.
        k (int, optional): Number of top predictions to display. Defaults to 10.
    """

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids

    # Forward pass
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits

    # Logits for the last token
    last_logits = logits[0, -1, :]

    # Convert to probabilities
    probs = torch.softmax(last_logits, dim=-1)

    # Get top "k" predictions
    topk = torch.topk(probs, k=10)
    topk_probs = topk.values
    topk_ids = topk.indices
    topk_tokens = [tokenizer.decode([token_id]) for token_id in topk_ids]

    # Display
    print(f"\nPrompt: {prompt!r}")
    print(f"\nTop {k} next-token predictions:\n")
    for token, prob in zip(topk_tokens, topk_probs):
        print(f"{token!r}: {prob.item():.4f}")


def get_embedding_gpt(model, tokenizer, input_text, layer_to_steer, normalize=False): # Warning! Do not normalize if get_steering_vector is used, only for singular embeddings
    """
    Get the mean-pooled embedding for a list of sentences from a specified layer of the model.
    
    Args:
        model: The causal language model.
        tokenizer: The tokenizer corresponding to the model.
        input_text (str or list): Text(s) to embed.
        layer_to_steer (int): Layer index to extract embeddings from.
        normalize (bool, optional): Whether to normalize embeddings. Defaults to False.
    
    Returns:
        torch.Tensor: Mean-pooled embedding vector.
    
    Warning:
        Do not normalize if using with get_steering_vector, only for singular embeddings.
    """

    if isinstance(input_text, str):
        input_text = [input_text]

    encoded_input = tokenizer(input_text, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        model_output = model(**encoded_input, output_hidden_states=True, return_dict=True)

    selected_layer_output = model_output.hidden_states[layer_to_steer] 

    pooled = mean_pooling((selected_layer_output,), encoded_input['attention_mask'])

    if normalize:
        pooled = F.normalize(pooled, p=2, dim=1)

    return pooled.mean(dim=0)



def get_steering_vector_gpt(model, tokenizer, feature, layer_to_steer, normalize=True):
    """
    Compute a steering vector for a single GPT-style feature directory.
    
    Args:
        model: The causal language model.
        tokenizer: The tokenizer corresponding to the model.
        feature (str): Feature name (directory containing feature.txt and optionally opposite.txt).
        layer_to_steer (int): Layer index to extract embeddings from.
        normalize (bool, optional): Whether to normalize the final steering vector. Defaults to True.
    
    Returns:
        torch.Tensor: Computed steering vector.
    
    Note:
        Expects 'feature.txt' and optionally 'opposite.txt' inside the Features/{feature} folder.
    """

    feature_texts, opposite_texts = import_feature_texts(f"../Features/{feature}")

    # Get embeddings for feature texts
    feature_embeddings = []
    for text in feature_texts:
        emb = get_embedding_gpt(model, tokenizer, text, layer_to_steer, normalize=False)
        feature_embeddings.append(emb)
    feature_vec = torch.stack(feature_embeddings).mean(dim=0)

    # If available, subtract opposite feature vector
    if opposite_texts:
        opposite_embeddings = []
        for text in opposite_texts:
            emb = get_embedding_gpt(model, tokenizer, text, layer_to_steer, normalize=False)
            opposite_embeddings.append(emb)
        opposite_vec = torch.stack(opposite_embeddings).mean(dim=0)
        feature_vec = feature_vec - opposite_vec

    if normalize:
        feature_vec = F.normalize(feature_vec, p=2, dim=0)

    return feature_vec



def steer_with_vector_gpt(model, tokenizer , prompt, layer_to_steer, steering_vector, steering_coefficient):
    """
    Steer the next token predictions of a model using a steering vector at a specified layer.
    
    Args:
        model: The causal language model.
        tokenizer: The tokenizer corresponding to the model.
        prompt (str): Input text prompt.
        layer_to_steer (int): Layer index to apply steering to.
        steering_vector (torch.Tensor): Steering vector to apply.
        steering_coefficient (float): Strength of the steering intervention.
    
    Returns:
        tuple: (logits, input_ids, model, tokenizer) - Model outputs and inputs.
    """

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids

    def steering_hook(module, input, output):
        return output + (steering_coefficient * steering_vector)

    handle = model.transformer.h[layer_to_steer].mlp.register_forward_hook(steering_hook)

    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits

    handle.remove()

    return logits, input_ids, model, tokenizer



def display_steered_next_tokens(model, tokenizer, prompt, layer_to_steer, steering_vector, steering_coefficient, k=10):
    """
    Display the next token predictions after steering the model with a steering vector.
    
    Args:
        model: The causal language model.
        tokenizer: The tokenizer corresponding to the model.
        prompt (str): Input text prompt.
        layer_to_steer (int): Layer index to apply steering to.
        steering_vector (torch.Tensor): Steering vector to apply.
        steering_coefficient (float): Strength of the steering intervention.
        k (int, optional): Number of top predictions to display. Defaults to 10.
    """

    logits, input_ids, _, tokenizer = steer_with_vector_gpt(model, tokenizer, prompt, layer_to_steer, steering_vector, steering_coefficient)

    # Get logits for the last token
    last_logits = logits[0, -1, :]

    # Convert to probabilities
    probs = torch.softmax(last_logits, dim=-1)

    # Top-k predictions
    topk = torch.topk(probs, k=k)
    topk_probs = topk.values
    topk_ids = topk.indices
    topk_tokens = [tokenizer.decode([token_id]) for token_id in topk_ids]

    # Print input and top predictions
    prompt_text = tokenizer.decode(input_ids[0])
    print(f"\nPrompt: {prompt_text!r}")
    print(f"Steering coefficient: {steering_coefficient}")
    print(f"\nTop {k} next-token predictions:\n")
    for token, prob in zip(topk_tokens, topk_probs):
        print(f"{token!r}: {prob.item():.4f}")
    

def generate_text(model, tokenizer, prompt, max_tokens=20, stop_token=".", temperature=1.0):
    """
    Generate text from a prompt using the specified model.

    Args:
        model: The causal language model.
        tokenizer: The tokenizer corresponding to the model.
        prompt (str): Input text prompt to continue.
        max_tokens (int, optional): Maximum number of tokens to generate. Defaults to 20.
        stop_token (str, optional): Token to stop generation. Defaults to ".".
        temperature (float, optional): Sampling temperature for randomness. Defaults to 1.0.
    
    Returns:
        str: Generated text based on prompt.
    
    Note:
        Temperature controls randomness: >1.0 more random, <1.0 more deterministic, →0 greedy.
        """
    
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    generated = input_ids.clone()

    model.eval()
    for _ in range(max_tokens):
        # Forward pass to get logits
        with torch.no_grad():
            outputs = model(generated)
            logits = outputs.logits

        # Get next token
        if temperature == 0:
            next_token_id = torch.argmax(logits[0, -1, :], dim=-1, keepdim=True)
        else:
            next_token_logits = logits[0, -1, :] / temperature
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token_id = torch.multinomial(probs, num_samples=1)

        # Append next token
        generated = torch.cat([generated, next_token_id.unsqueeze(0)], dim=1)

        # Decode and check for stop_token
        decoded_token = tokenizer.decode(next_token_id[0])
        if stop_token and decoded_token.strip() == stop_token:
            break

    # Final decode
    return tokenizer.decode(generated[0], skip_special_tokens=True)



def generate_steered_text(model, tokenizer, prompt, layer_to_steer, steering_vector, steering_coefficient,max_tokens=20, stop_token=".", temperature=1.0):
    """
    Generate text from a prompt by applying a steering vector at a specific layer.
    
    Args:
        model: The causal language model.
        tokenizer: The tokenizer corresponding to the model.
        prompt (str): Input text prompt to continue.
        layer_to_steer (int): Layer index to apply steering to.
        steering_vector (torch.Tensor): Steering vector to apply.
        steering_coefficient (float): Strength of the steering intervention.
        max_tokens (int, optional): Maximum number of tokens to generate. Defaults to 20.
        stop_token (str, optional): Token to stop generation. Defaults to ".".
        temperature (float, optional): Sampling temperature for randomness. Defaults to 1.0.
    
    Returns:
        str: Generated text with steering applied.
    
    Note:
        Temperature controls randomness: >1.0 more random, <1.0 more deterministic, →0 greedy.
    """
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    generated = input_ids.clone()

    model.eval()
    for _ in range(max_tokens):
        # Steer model using current input
        prompt_text = tokenizer.decode(generated[0], skip_special_tokens=True)
        logits, _, _, _ = steer_with_vector_gpt(model, tokenizer, prompt_text, layer_to_steer, steering_vector, steering_coefficient)

        # Get next token
        if temperature == 0:
            next_token_id = torch.argmax(logits[0, -1, :], dim=-1, keepdim=True)
        else:
            next_token_logits = logits[0, -1, :] / temperature
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token_id = torch.multinomial(probs, num_samples=1)

        # Append next token
        generated = torch.cat([generated, next_token_id.unsqueeze(0)], dim=1)

        # Decode and check for stop_token
        decoded_token = tokenizer.decode(next_token_id[0])
        if stop_token and decoded_token.strip() == stop_token:
            break

    # Final decode
    return tokenizer.decode(generated[0], skip_special_tokens=True)






"""
Temperature	Behavior
1.0 (default)	        Standard sampling. Balance between randomness and likelihood.
>1.0 (e.g. 1.5, 2.0)	Flatter distribution → more randomness, more surprising/creative words.
<1.0 (e.g. 0.7, 0.3)	Sharper distribution → less randomness, favors high-probability tokens.
→ 0	                    Approaches greedy decoding — always picks the highest-probability token.

"""




if __name__ == "__main__":
    # Example usage

    model_name = "openai-community/gpt2"
    prompt = "I am a woman, my doctor is a"
    model, tokenizer = initialize_model_and_tokenizer(model_name)

    # Example of strings: Women
    steering_sentences = [
    "She braided her daughter’s hair with one hand while sending an email with the other — and no one questioned it.",
    "The midwife stood calm as storms, her voice steadier than the monitors beeping beside her.",
    "Wearing heels or combat boots, she walks like the world owes her space — and it does.",
    "She bleeds monthly and still runs marathons, meetings, and entire households.",
    "The senator adjusted her blazer and silenced the room before saying a single word.",
    "She is the matriarch, the memory-keeper, the one everyone calls when things fall apart.",
    "Her lipstick is warpaint, and her silence is strategy.",
    "From nursery rhymes to protest chants, her voice has always carried more than melody.",
    "She stitched every family story into the quilt that now warms three generations.",
    "She grew life inside her, lost sleep for years, and still built a business from scratch.",
    "The grandmother who crossed borders with babies strapped to her chest — that’s who she is.",
    "She is the girl told to smile, the teen told to shrink, the woman who refused.",
    "Behind every medal, there's a ponytail soaked in sweat and defiance.",
    "She signs her name where others once wrote hers for her.",
    "You can find her in every history book margin — not because she wasn’t there, but because someone tried to erase her."
    ]

    layer_to_steer = 10
    steering_coefficient = 3

    display_next_tokens(model, tokenizer, prompt)

    steering_vector = get_embedding_gpt(model, tokenizer, steering_sentences, layer_to_steer, normalize=False)

    display_steered_next_tokens(model, tokenizer, prompt, layer_to_steer, steering_vector, steering_coefficient, k=10)

