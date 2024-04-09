# %%
device = "cpu"


# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import matplotlib.pyplot as plt
import tqdm.notebook as tqdm


# %%
transforms = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Resize((64, 64)),
    ]
)

dataset = torchvision.datasets.ImageFolder(
    "celeb-hq/data/celeba_hq/train", transform=transforms
)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

x, y = next(iter(dataloader))
print(x.shape, y.shape)
print(dataset.classes)


# %%
def gradient_penalty(
    discriminator, real: torch.Tensor, fake: torch.Tensor
) -> torch.Tensor:
    alpha = torch.rand(real.size(0), 1, 1, 1).to(real.device)
    interpolates = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
    d_interpolates = discriminator(interpolates)
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones(d_interpolates.size()).to(real.device),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


# DCGAN


d = 8

generator = Generator(d).to(device)
discriminator = Discriminator(d).to(device)

print(
    f"Generator has {sum(p.numel() for p in generator.parameters() if p.requires_grad):,} parameters"
)
print(
    f"Discriminator has {sum(p.numel() for p in discriminator.parameters() if p.requires_grad):,} parameters"
)


# %%
optim_d = optim.Adam(discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999))
optim_g = optim.Adam(generator.parameters(), lr=2e-4, betas=(0.5, 0.999))


# %%
with tqdm.tnrange(10) as pbar:
    for epoch in pbar:
        pbar.set_description(f"Epoch {epoch}")
        with tqdm.tqdm_notebook(
            dataloader, desc=f"Epoch {epoch}", ascii=True
        ) as pbatch:
            for x, _ in pbatch:
                x = x.to(device)
                z = torch.randn(x.size(0), 100).to(device)

                # Train discriminator
                optim_d.zero_grad()
                d_real = discriminator(x)
                d_fake = discriminator(generator(z).detach())
                loss_d = (
                    d_fake.mean()
                    - d_real.mean()
                    + gradient_penalty(discriminator, x, generator(z))
                    + 0.001 * (d_real**2).mean()
                )
                loss_d.backward()
                optim_d.step()

                # Train generator
                optim_g.zero_grad()
                d_fake = discriminator(generator(z))
                loss_g = -d_fake.mean()
                loss_g.backward()
                optim_g.step()

                pbatch.set_description(f"Epoch {epoch}, Batch {pbatch.n}")
                pbatch.set_postfix_str(
                    f"Loss D: {loss_d.item():.2f}, Loss G: {loss_g.item():.2f}, Sliced W: {d_real.mean().item() - d_fake.mean().item():.2f}, D Real: {d_real.mean().item():.2f}, D Fake: {d_fake.mean().item():.2f}"
                )


# %%
noise = torch.randn(16**2, 100, device=device)
fake = generator(noise)
plt.figure(figsize=(20, 20))
grid = torchvision.utils.make_grid(fake, nrow=16).clamp(0, 1)
plt.imshow(grid.permute(1, 2, 0).detach().cpu().numpy())
plt.axis("off")
plt.show()


# %%
