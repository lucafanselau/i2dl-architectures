from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import torchvision
from torch.utils.hooks import RemovableHandle

writer = SummaryWriter()


def write_feature_maps(model, val_loader, device):
    feature_maps = []

    def get_features(name):
        def hook(model, input, output):
            feature_maps.append(output)

        return hook

    handles: list[RemovableHandle] = []
    for layer in model.features:
        if isinstance(layer, nn.Conv2d):
            handles.append(layer.register_forward_hook(get_features(layer)))

    images, labels = next(iter(val_loader))  # Get the first batch of images
    images = images.to(device)
    model(images[0].unsqueeze(0))  # Pass the first image through the model

    # upload image to tensorboard
    grid = torchvision.utils.make_grid(images[0])
    writer.add_image("images", grid, 0)

    # also upload the labels
    writer.add_text("labels", str(labels[0].item()), 0)

    for i, feature_map in enumerate(feature_maps):
        # normalize feature map to 0..1 range
        feature_map = (feature_map - feature_map.min()) / (
            feature_map.max() - feature_map.min()
        )
        feature_map = (feature_map * 255).byte()
        # rechape the tensor with shape (1, C, W, H) to a tensor with shape (M, 3, W, H)
        w, h = feature_map.shape[-2:]
        feature_map = feature_map.view(-1, 1, w, h).expand(-1, 3, -1, -1)
        # detach from graph and convert to numpy
        feature_map = feature_map.detach().cpu().numpy()

        writer.add_images(f"Feature Map {i}", feature_map, 0)

    writer.close()
    for handle in handles:
        handle.remove()
