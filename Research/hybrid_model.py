import torch
import torch.nn as nn

class MockMambaBlock(nn.Module):
    """
    Simulates a Selective State Space linear scan O(N).
    (In full production, replace this with: from mamba_ssm import Mamba)
    """
    def __init__(self, d_model):
        super().__init__()
        self.linear_proj = nn.Linear(d_model, d_model)
        self.conv1d = nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=3, padding=1)
        self.silu = nn.SiLU()

    def forward(self, x):
        # x shape: [batch, seq_len, dim]
        x_t = x.transpose(1, 2) # Conv1d expects [batch, dim, seq_len]
        h = self.conv1d(x_t)
        h = h.transpose(1, 2)
        return self.silu(self.linear_proj(x) + h)


class ResumeHybridEngine(nn.Module):
    """
    Interleaves Mamba linear digestion with Transformer Self-Attention.
    """
    def __init__(self, d_model=768, n_heads=8):
        super().__init__()
        self.d_model = d_model
        
        # 1. Linear Complex Layer: Fast document history compression
        self.mamba_layer = MockMambaBlock(d_model)
        self.norm1 = nn.LayerNorm(d_model)
        
        # 2. Quadratic Complex Layer: High-precision needle retrieval
        self.attention_layer = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        
        # 3. Schema Projection Layer
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model)
        )

    def forward(self, token_embeddings):
        # Step 1: Pass through Mamba (O(N))
        residual = token_embeddings
        out = self.norm1(token_embeddings)
        mamba_out = self.mamba_layer(out) + residual
        
        # Step 2: Pass through Attention (O(N^2))
        residual = mamba_out
        out = self.norm2(mamba_out)
        attn_out, _ = self.attention_layer(out, out, out)
        attn_out = attn_out + residual
        
        # Step 3: Feature Map
        final_representations = self.ffn(attn_out)
        return final_representations

# --- THE LIVE TENSOR VERIFICATION ---
if __name__ == "__main__":
    print("🧠 Initializing Hybrid Mamba-Transformer Tensor Architecture...")
    model = ResumeHybridEngine(d_model=768, n_heads=8)
    
    # Simulate a batch of 1 resume containing 1024 embedded tokens
    dummy_resume_tokens = torch.randn(1, 1024, 768)
    
    print(f"📥 Input Tensor Shape:  {dummy_resume_tokens.shape}")
    output_tensor = model(dummy_resume_tokens)
    print(f"📤 Output Tensor Shape: {output_tensor.shape}")
    
    assert dummy_resume_tokens.shape == output_tensor.shape
    print("✅ Forward pass successful! Dimensions strictly preserved.")
