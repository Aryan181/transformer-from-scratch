import numpy as np  

word_embeddings = [] 
position_embeddings = [] 
sentence = "the dog sat"
words = sentence.split()  
final_embeddings = []

queries = []
keys = []
values = []


for i in range(len(words)):
    vector_4d = np.random.rand(4)  
    
    pos_vector = np.array([np.sin(i), np.cos(i), np.sin(i), np.cos(i)])

    combined_vectors = vector_4d + pos_vector

    word_embeddings.append(vector_4d)
    final_embeddings.append(combined_vectors)


matrix = np.array(final_embeddings)


W_Q = np.random.rand(4, 4)
W_K = np.random.rand(4, 4)
W_V = np.random.rand(4, 4)


for embedding in final_embeddings: 
    queries.append(embedding @ W_Q)
    keys.append(embedding @ W_K)
    values.append(embedding @ W_V)


def softmax(x):
    e_x = np.exp(x - np.max(x))  # subtract max for numerical stability
    return e_x / e_x.sum()


attention_outputs = []

for query in queries:

    scores = []

    for key in keys:
        dot = np.dot(query, key)
        scaled = dot / 2.0
        scores.append(scaled)

    weights = softmax(scores)

    final_vector = final_vector = np.sum([w * v for w, v in zip(weights, values)], axis=0)
    attention_outputs.append(final_vector)

print(np.array(attention_outputs))


W1 = np.random.rand(4, 8)
W2 = np.random.rand(8, 4)

def relu(x):
    return np.maximum(0, x)

ffn_outputs = []

for i in range(len(attention_outputs)):
    residual = attention_outputs[i] + final_embeddings[i]   
    hidden = relu(residual @ W1)  
    final = hidden @ W2          
    output = final + residual     
    ffn_outputs.append(output)

print(np.array(ffn_outputs))


 
vocab = ["on", "down", "quickly", "the"]
vocab_size = len(vocab)

 
W_out = np.random.rand(4, vocab_size)

 
last_vector = ffn_outputs[-1]   
 
logits = last_vector @ W_out   

 
probs = softmax(logits)

# Print vocab with probabilities
for word, prob in zip(vocab, probs):
    print(f"{word}: {prob:.3f}")

# Optional: get most likely next word
next_word = vocab[np.argmax(probs)]
print("\nPredicted next word:", next_word)
