from pathlib import Path
import torch as t

ids = ["1926876", "1997481", "2006869", "2016869"]
embedding_folder = Path("saved/")
output_folder = Path("saved_sorted/")
data_folder = Path("F:/MLiPHotel-IDData/Hotel-ID-2022/")
for saved_id in ids:
    base_embeddings = t.load(embedding_folder / f'{saved_id}_embeds.pt')
    base_hids = t.load(embedding_folder / f'{saved_id}_hids.pt')

    filenames = list((data_folder / "train_images").glob("**/*.jpg"))
    sorted_filenames = sorted(enumerate(filenames), key=lambda x: x[1])
    
    indices, sorted_filenames = zip(*sorted_filenames)
    indices = list(indices)
    new_base_embeddings = base_embeddings[indices,:]
    new_base_hids = base_hids[indices]

    if t.equal(new_base_embeddings, base_embeddings):
        print(f"Base embeddings for {saved_id} are the same")
    else:
        t.save(new_base_embeddings, output_folder / f'{saved_id}_embeds.pt')
    if t.equal(new_base_hids, base_hids):
        print(f"Base hids for {saved_id} are the same")
    else:
        t.save(new_base_hids, embedding_folder / f'{saved_id}_hids.pt')