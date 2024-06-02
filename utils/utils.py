import torch


def save_state_dict(model, optimizer, epoch, model_name, checkpoint_dir):
    """
    Save the state of a model and its optimizer.

    Args:
        model (torch.nn.Module): The model to save.
        optimizer (torch.optim.Optimizer): Optimizer for the model.
        epoch (int): Current epoch number.
        model_name (str): Name of the model ('generator' or 'discriminator').
        checkpoint_dir (str): Directory to save the model checkpoint.
    """
    model_path = f"{checkpoint_dir}/{model_name}_epoch_{epoch}.pth"
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, model_path)
    print(f"{model_name.capitalize()} model saved successfully at epoch {epoch}. Checkpoint stored in {model_path}")


def save_checkpoints(generator, discriminator, optimizerG, optimizerD, epoch, checkpoint_dir):
    save_state_dict(generator, optimizerG, epoch, 'generator', checkpoint_dir)
    save_state_dict(discriminator, optimizerD, epoch, 'discriminator', checkpoint_dir)
