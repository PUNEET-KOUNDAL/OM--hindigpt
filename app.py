import streamlit as st
import torch
import torch.nn.functional as F

# Load model components from your script
# (You might want to refactor some things into a separate file for cleaner code)

# Same character setup
with open(r'D:\PROJECTS\om-gpt\archive\input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Define the model class (same as before)
import torch.nn as nn

class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        logits = self.token_embedding_table(idx)
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, loss = self(idx)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# Load trained model (make sure to save it after training in your main script)
model = BigramLanguageModel(vocab_size)
model.load_state_dict(torch.load("bigram_model.pth", map_location=device))
model.to(device)
model.eval()

# Streamlit UI
import streamlit as st
import torch

# Assuming `encode`, `decode`, `model`, and `device` are already defined
# For example, your model might look like this:
# model = YourModel().to(device)

st.title("💬 HINDI-GPT")

# Input field for user to enter Hindi text
user_input = st.text_input("हिंदी में कुछ टाइप करें", "")

if st.button("Generate"):
    if user_input:
        # Encoding the user input and passing it to the model
        context = torch.tensor([encode(user_input)], dtype=torch.long).to(device)
        output = model.generate(context, max_new_tokens=100)
        # Decoding the model's output
        generated_text = decode(output[0].tolist())
        st.text_area("Generated Response", generated_text, height=200)
    else:
        st.warning("Please enter some text!")

# Adding a footer message
st.markdown(
    """
    <style>
    .footer {
        position: absolute;
        bottom: 0;
        left: 0;
        right: 0;
        padding: 10px;
        text-align: center;
        font-size: 12px;
        color: grey;
    }
    </style>
    <div class="footer">
        This is just a prototype. We trained this model on limited data, which is why it might be inefficient.
    </div>
    """,
    unsafe_allow_html=True
)
