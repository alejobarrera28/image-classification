import os
import shutil


def organize_tiny_imagenet():
    """Convert messy Tiny ImageNet to clean structure"""

    # Paths
    raw_dir = "data/raw/tiny-imagenet-200"
    processed_dir = "data/preprocessed"

    # 1. Training data
    train_src = f"{raw_dir}/train"
    train_dst = f"{processed_dir}/train"

    if not os.path.exists(train_dst):
        print("Organizing training data...")
        os.makedirs(train_dst)

        for class_dir in os.listdir(train_src):
            class_src_path = f"{train_src}/{class_dir}"
            class_dst_path = f"{train_dst}/{class_dir}"
            os.makedirs(class_dst_path)

            images_src_path = f"{class_src_path}/images"
            if os.path.exists(images_src_path):
                for image_file in os.listdir(images_src_path):
                    if image_file.endswith(".JPEG"):
                        src_file = f"{images_src_path}/{image_file}"
                        dst_file = f"{class_dst_path}/{image_file}"
                        shutil.copy2(src_file, dst_file)

    # 2. Validation data
    val_src_images = f"{raw_dir}/val/images"
    val_annotations = f"{raw_dir}/val/val_annotations.txt"
    val_dst = f"{processed_dir}/val"

    if not os.path.exists(val_dst):
        print("Organizing validation data...")
        os.makedirs(val_dst)

        image_to_class = {}
        with open(val_annotations, "r") as f:
            for line in f:
                parts = line.strip().split("\t")
                image_name = parts[0]
                class_id = parts[1]
                image_to_class[image_name] = class_id

        for image_name, class_id in image_to_class.items():
            class_folder = f"{val_dst}/{class_id}"
            os.makedirs(class_folder, exist_ok=True)

            src_image = f"{val_src_images}/{image_name}"
            dst_image = f"{class_folder}/{image_name}"
            shutil.copy2(src_image, dst_image)

    # 3. Test data
    test_src_images = f"{raw_dir}/test/images"
    test_dst = f"{processed_dir}/test"

    if not os.path.exists(test_dst):
        print("Organizing test data...")
        os.makedirs(test_dst)

        for image_file in os.listdir(test_src_images):
            if image_file.endswith(".JPEG"):
                src_file = f"{test_src_images}/{image_file}"
                dst_file = f"{test_dst}/{image_file}"  # Flat structure!
                shutil.copy2(src_file, dst_file)

    # 4. Create human-readable class mappings
    mappings_dir = "data/class_mappings"
    os.makedirs(mappings_dir, exist_ok=True)

    # Copy WordNet IDs
    shutil.copy2(f"{raw_dir}/wnids.txt", f"{mappings_dir}/wnids.txt")

    # Create human-readable names (if words.txt exists)
    words_file = f"{raw_dir}/words.txt"
    if os.path.exists(words_file):
        shutil.copy2(words_file, f"{mappings_dir}/words.txt")

        # Create clean mapping file with just Tiny ImageNet classes
        print("Creating clean class names mapping...")

        with open(f"{mappings_dir}/wnids.txt", "r") as f:
            tiny_classes = set(line.strip() for line in f)

        with open(f"{mappings_dir}/class_names.txt", "w") as out_file:
            with open(f"{mappings_dir}/words.txt", "r") as in_file:
                for line in in_file:
                    parts = line.strip().split("\t")
                    if len(parts) >= 2:
                        wnid = parts[0]
                        full_name = parts[1]

                        if wnid in tiny_classes:
                            clean_name = full_name.split(",")[0].strip()
                            out_file.write(f"{wnid}\t{clean_name}\n")


    print("\nTiny ImageNet preprocessing complete")
    print("=" * 60)

    # Training stats
    train_class_dirs = [
        d
        for d in os.listdir(train_dst)
        if d.startswith("n") and len(d) == 9 and d[1:].isdigit()
    ]
    train_classes = len(train_class_dirs)
    train_images_per_class = (
        len(
            [
                f
                for f in os.listdir(f"{train_dst}/{train_class_dirs[0]}")
                if f.endswith(".JPEG")
            ]
        )
        if train_class_dirs
        else 0
    )
    print(
        f"Training: {train_classes} classes, {train_images_per_class} images per class. Total {train_classes * train_images_per_class} images"
    )

    # Validation stats
    val_class_dirs = [
        d
        for d in os.listdir(val_dst)
        if d.startswith("n") and len(d) == 9 and d[1:].isdigit()
    ]
    val_classes = len(val_class_dirs)
    val_images_per_class = (
        len(
            [
                f
                for f in os.listdir(f"{val_dst}/{val_class_dirs[0]}")
                if f.endswith(".JPEG")
            ]
        )
        if val_class_dirs
        else 0
    )
    print(
        f"Validation: {val_classes} classes, {val_images_per_class} images per class. Total {val_classes * val_images_per_class} images"
    )

    # Test stats (just total images, no classes)
    test_images = len(os.listdir(test_dst))
    print(f"Test: {test_images} images (unlabeled)")


if __name__ == "__main__":
    organize_tiny_imagenet()
