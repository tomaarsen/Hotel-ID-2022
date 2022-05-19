
# import sys
# sys.path.insert(0, "..")

from matplotlib import pyplot as plt
from torchvision import transforms
from PIL import Image
from pathlib import Path
from skeleton.data.dataset import ImageDataset
from skeleton.data.preprocess import Preprocessor

data_folder = Path("F:\MLiPHotel-IDData\Hotel-ID-2022")

# Get the list of hotel_ids, i.e. a mapping of index/label to hotel ID
hotel_ids = sorted(
    int(folder.stem) for folder in (data_folder / "train_images").iterdir()
)

processor = Preprocessor(512, 512)

image_folder = data_folder / "train_images"
filepaths = list(image_folder.glob("**/*.jpg"))
ds = ImageDataset(filepaths, hotel_ids, lambda image=None: {"image": image})

to_image = transforms.ToPILImage()
for sample in ds:
    # Create 2 subplots, one for loss, one for accuracy
    fig, (before_ax, after_ax) = plt.subplots(1, 2)
    fig.suptitle(f"Before and after applying transformations")

    # Before subplot
    before_ax.imshow(sample.image, label="Training Loss")

    # After subplot
    augmented = processor.test_transform(image=sample.image)["image"]
    after_ax.imshow(to_image(augmented), label="Training Loss")

    fig.tight_layout()
    plt.show()