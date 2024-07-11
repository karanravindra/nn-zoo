import torch
from nn_zoo.models import VQVAE


def test_vqvae():
    model = VQVAE(
        in_channels=3,
        out_channels=3,
        num_hiddens=32,
        num_downsampling_layers=2,
        num_residual_layers=2,
        num_residual_hiddens=32,
        embedding_dim=64,
        num_embeddings=512,
        use_ema=True,
        decay=0.99,
        epsilon=1e-5,
    )
    assert model
    assert model.encoder
    assert model.decoder

    x = torch.rand(1, 3, 32, 32)
    y = model(x)
    assert y["x_recon"].shape == (1, 3, 32, 32)


if __name__ == "__main__":
    test_vqvae()
