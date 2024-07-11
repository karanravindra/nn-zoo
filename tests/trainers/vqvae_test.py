from nn_zoo.trainers import VQVAE_Trainer


def vqvae_trainer():
    model = VQVAE_Trainer()
    assert model


if __name__ == "__main__":
    vqvae_trainer()
    print("VQVAE Trainer test passed")
