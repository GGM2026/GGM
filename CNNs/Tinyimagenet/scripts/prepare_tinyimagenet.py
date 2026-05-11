import os
import wget
import zipfile
import sys

def prepare_tiny_imagenet(root='./data'):
    url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
    target_dir = os.path.join(root, 'tiny-imagenet-200')
    
    if os.path.exists(target_dir):
        print(f"Dataset already exists at {target_dir}")
        return

    
    os.makedirs(root, exist_ok=True)

    print(f"Downloading TinyImageNet to {root}...")
    filename = wget.download(url, out=root)
    print("\nUnzipping...")
    with zipfile.ZipFile(filename, 'r') as zip_ref:
        zip_ref.extractall(root)
    
    
    val_dir = os.path.join(target_dir, 'val')
    img_dir = os.path.join(val_dir, 'images')
    
    fp = open(os.path.join(val_dir, 'val_annotations.txt'), 'r')
    data = fp.readlines()
    
    val_img_dict = {}
    for line in data:
        words = line.split('\t')
        val_img_dict[words[0]] = words[1]
    fp.close()

    print("Restructuring validation set...")
    for img, folder in val_img_dict.items():
        new_dir = os.path.join(val_dir, folder)
        os.makedirs(new_dir, exist_ok=True)
        old_path = os.path.join(img_dir, img)
        new_path = os.path.join(new_dir, img)
        if os.path.exists(old_path):
            os.rename(old_path, new_path)
            

    os.rmdir(img_dir)
    os.remove(filename)
    print("Done!")

if __name__ == "__main__":
   
    data_root = sys.argv[1] if len(sys.argv) > 1 else './data'
    prepare_tiny_imagenet(data_root)