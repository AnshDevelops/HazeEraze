import time
import torch
from utils.utils import save_checkpoints


def train_cycle_gan(generator, discriminator, dataloader, optimizerG, optimizerD, criterionGAN, criterionCycle,
                    criterionIdentity, perceptual_loss, device, epochs, save_interval, checkpoint_dir):
    """
    Train CycleGAN models with given dataloader and optimizers.

    Args:
        generator (torch.nn.Module): The generator model.
        discriminator (torch.nn.Module): The discriminator model.
        dataloader (torch.utils.data.DataLoader): DataLoader for the dataset.
        optimizerG (torch.optim.Optimizer): Optimizer for the generator.
        optimizerD (torch.optim.Optimizer): Optimizer for the discriminator.
        criterionGAN (torch.nn.Module): GAN loss function.
        criterionCycle (torch.nn.Module): Cycle consistency loss function.
        criterionIdentity (torch.nn.Module): Identity loss function.
        perceptual_loss (torch.nn.Module): Perceptual loss function.
        device (torch.device): Device to run the model computation.
        epochs (int): Number of epochs to train.
        save_interval (int): Interval to save the model checkpoint.
        checkpoint_dir (str): Directory to save the model checkpoints.
    """
    generator.to(device)
    discriminator.to(device)

    for epoch in range(epochs):
        start_time = time.time()
        for i, data in enumerate(dataloader):
            real_images, labels = data['hazy'].to(device), data['clear'].to(device)

            # Discriminator training
            optimizerD.zero_grad()
            real_output = discriminator(real_images)
            lossD_real = criterionGAN(real_output, torch.ones_like(real_output))
            fake_images = generator(real_images)
            fake_output = discriminator(fake_images.detach())
            lossD_fake = criterionGAN(fake_output, torch.zeros_like(fake_output))
            lossD = (lossD_real + lossD_fake) / 2
            lossD.backward()
            optimizerD.step()

            # Generator training
            optimizerG.zero_grad()
            fake_output = discriminator(fake_images)
            lossG_GAN = criterionGAN(fake_output, torch.ones_like(fake_output))
            recovered_images = generator(fake_images)
            lossG_Cycle = criterionCycle(recovered_images, real_images) * 10.0
            identity_images = generator(labels)
            lossG_Identity = criterionIdentity(identity_images, labels) * 5.0
            lossG_Perceptual = perceptual_loss(fake_images, labels) * 2.0
            lossG = lossG_GAN + lossG_Cycle + lossG_Identity + lossG_Perceptual
            lossG.backward()
            optimizerG.step()

            if i % 50 == 0:
                print(f"Epoch [{epoch + 1}/{epochs}], Step [{i}/{len(dataloader)}], "
                      f"D Loss: {lossD.item():.4f}, G Loss: {lossG.item():.4f}")

        if (epoch + 1) % save_interval == 0:
            save_checkpoints(generator, discriminator, optimizerG, optimizerD, epoch, checkpoint_dir)

        print(f"Epoch [{epoch + 1}/{epochs}] completed in {(time.time() - start_time):.2f} seconds")
