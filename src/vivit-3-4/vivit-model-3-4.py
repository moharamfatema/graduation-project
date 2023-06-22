import einops
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class FeedForwardBlock(tf.Module):
    def __init__(self, embed_dim, hidden_dim, dropout=0.):
        self.net = keras.Sequential(
            [
                layers.Linear(units=hidden_dim),
                tf.nn.gelu,
                layers.Dropout(rate=dropout),
                layers.Linear(units=embed_dim),
                layers.Dropout(rate=dropout),
            ])

    def __call__(self, inputs):
        x = self.net(inputs)
        return x


class FSAttention(tf.Module):
    def __init__(self, embed_dim, dim_head, num_heads, dropout=0.0):
        self.embed_dim = embed_dim
        self.inner_dim = dim_head * num_heads
        self.dim_head = dim_head
        self.num_heads = num_heads
        self.dropout = dropout
        project_out = not (num_heads == 1 and dim_head == embed_dim)

        self.scale = dim_head ** -0.5
        self.to_qkv = layers.linear(embed_dim * 3, use_bias=False)
        self.to_out = keras.Sequential(
            [
                layers.linear(embed_dim),
                layers.Dropout(dropout),
            ]) if project_out else tf.identity

    def __call__(self, x):
        _, _, _, h = *x.shape, self.num_heads
        qkv = self.to_qkv(x).reshape(*x.shape[:-1], 3, h, self.dim_head)
        q, k, v = tf.unstack(qkv, axis=-3)
        dots = tf.einsum("...ndh,...mdh->...nmh", q, k) * self.scale
        attn = dots.softmax(dim=-1)
        out = tf.einsum("...nmh,...mdh->...ndh", attn, v)
        out = out.reshape(*x.shape[:-1], self.inner_dim)
        return self.to_out(out)

class FSATransformerBlock(tf.Module):
    def __init__(self, embed_dim, depth, num_heads, mlp_dim,nt,nh,nw, dropout=0.0):
        self.layers = []
        self.nt = nt
        self.nh = nh
        self.nw = nw

        for _ in range(depth):
            self.layers.append(tf.ModuleList([
                layers.LayerNorm(epsilon=1e-6),
                FSAttention(embed_dim, embed_dim // num_heads, num_heads, dropout),
                layers.layerNorm(epsilon=1e-6),
                FSAttention(embed_dim, embed_dim // num_heads, num_heads, dropout),
                layers.LayerNorm(epsilon=1e-6),
                FeedForwardBlock(embed_dim, mlp_dim, dropout),
            ]))

    def __call__(self, x):
        b = x.shape[0]
        x = x.flatten(2)
        for _, attn1,_, attn2,_, ff in self.layers:
            x1 = x + attn1(x) # spatial attention
            x1 = x1.reshape(b,self.nt,self.nh,self.nw,0)
            x1 = [temp[None] for temp in x1]
            x1 = np.concatenate(x1,axis=0).transpose(1,2)

            x2 = x1 + attn2(x1) # temporal attention
            x = ff(x2) + x2 # MLP

            # reshape tensor for spatial attention
            x = x.reshape(b,self.nt,self.nh,self.nw,0)
            x = [temp[None] for temp in x]
            x = np.concatenate(x,axis=0).transpose(1,2)
            x = x.flatten(2)

        x = x.reshape((b,self.nt*self.nh*self.nw,-1))
        x = [temp[None] for temp in x]
        x = np.concatenate(x,axis=0).flatten(2)
        return x
        
class FDAttention(tf.Module):
  def __init__(self, dim, nt, nh, nw, heads=8, dim_head=64, dropout=0.0):
    inner_dim = dim_head * heads
    self.nt = nt
    self.nh = nh
    self.nw = nw
    self.heads = heads
    self.scale = dim_head ** -0.5
    project_out = not (heads == 1 and dim_head == dim)
    self.attend = layers.Softmax(axis=-1)
    self.to_qkv = layers.linear(inner_dim * 3, use_bias=False)
    self.to_out = layers.Sequential(
            [
                layers.linear(inner_dim),
                layers.Dropout(dropout),
            ]) if project_out else tf.identity
    def forward(self, x):
        b, n, d, h = *x.shape, self.heads
        qkv = self.to_qkv(x).reshape(*x.shape[:-1], 3, h, self.dim_head)      
        q, k, v = tf.unstack(qkv, axis=-3)
        qs, qt = tf.split(q,2,axis =1)
        ks, kt = tf.split(k,2,axis =1)
        vs, vt = tf.split(v,2,axis =1)

        qs = tf.reshape(qs,[b, h // 2, self.nt, self.nh * self.nw])
        ks = tf.reshape(ks,[b, h // 2, self.nt, self.nh * self.nw])
        vs = tf.reshape(vs,[b, h // 2, self.nt, self.nh * self.nw])

        spatial_dots = tf.einsum("...ndh,...mdh->...nmh", qs, ks) * self.scale
        sp_attn = self.attend(spatial_dots)
        spatial_out = tf.einsum("...nmh,...mdh->...ndh", sp_attn, vs)

        qt = tf.reshape(qt,[b, h // 2, self.nh, self.nw * self.nt])
        kt = tf.reshape(kt,[b, h // 2, self.nh, self.nw * self.nt])
        vt = tf.reshape(vt,[b, h // 2, self.nh, self.nw * self.nt])

        temporal_dots = tf.einsum("...ndh,...mdh->...nmh", qt, kt) * self.scale
        temporal_attn = self.attend(temporal_dots)
        temporal_out = tf.einsum("...nmh,...mdh->...ndh", temporal_attn, vt)
class FDATransformerEncoder(tf.Module):
  def __init__(self, dim, depth, heads, dim_head, mlp_dim, nt, nh, nw, dropout=0.0):
        self.layers = []
        self.nt = nt
        self.nh = nh
        self.nw = nw

        for _ in range(depth):
            self.layers.append(tf.ModuleList([
                layers.LayerNorm(epsilon=1e-6),
                FDAttention(dim, nt, nh, nw, heads=heads, dim_head=dim_head, dropout=dropout)
            ]))
  def forward(self, x):
        for _, attn in self.layers:
            x = attn(x) + x

        return x         

class VIVITBAckbone(tf.Module):
    def __init__(self, t, h, w, patch_t, patch_h, patch_w, num_classes, embed_dim, depth, num_heads, mlp_dim, dim_head = 3, dropout=0.0, channels=3, emb_dropout=0.0):
        assert (t, h, w) % (patch_t, patch_h, patch_w) == 0, "Image dimensions must be divisible by the patch size."

        self.patch_t = patch_t
        self.patch_h = patch_h
        self.patch_w = patch_w
        self.channels = channels
        self.t = t
        self.h = h
        self.w = w

        self.nt = t // patch_t
        self.nh = h // patch_h
        self.nw = w // patch_w

        tubelet_dim = (patch_t * patch_h * patch_w) * channels
        self_to_tubelet = layers.Sequential(
            [
                einops.rearrange('b c (t pt) (h ph) (w pw) -> b t (h w) (pt ph pw c)', pt=patch_t, ph=patch_h, pw=patch_w),
                layers.linear(embed_dim),
            ]
        )
        self.pos_embedding = layers.Parameter(tf.random.normal((1,1, self.nh * self.nw, embed_dim)).repeat(1,self.nt,1,1))
        self.dropout = layers.Dropout(emb_dropout)
        assert num_heads % 2 == 0, "num_heads must be divisible by 2."
        self.transformer = FSATransformerBlock(embed_dim, depth, num_heads, mlp_dim, self.nt, self.nh, self.nw, dropout)
        self.to_latent = layers.Identity()
        self.mlp_head = keras.Sequential(
            [
                layers.LayerNorm(epsilon=1e-6),
                layers.linear(num_classes),
            ]
        )

    def __call__(self, x):
        tokens = self.self_to_tubelet(x)
        tokens = tokens + self.pos_embedding
        tokens = self.dropout(tokens)
        x = self.transformer(tokens)
        x = x.mean(dim=1)
        x = self.to_latent(x)
        return self.mlp_head(x)
