import os
from pdf2image import convert_from_path

DATASET_PATH = "./dataset"

def split_and_convert_pdfs(base_path):
    # Iterate through 'train' and 'val' folders
    for split in ['train', 'val']:
        split_path = os.path.join(base_path, split)

        for category in os.listdir(split_path):
            category_path = os.path.join(split_path, category)

            if not os.path.isdir(category_path):
                continue

            print(f"Processing Category: {category} in {split}")

            for filename in os.listdir(category_path):
                if filename.lower().endswith(".pdf"):
                    pdf_path = os.path.join(category_path, filename)

                    # Comvert PDF to List of Images
                    pages = convert_from_path(pdf_path, dpi=200)

                    for i, page in enumerate(pages):
                        new_filename = f"{os.path.splitext(filename)[0]}_page_{i+1}.jpg"
                        save_path = os.path.join(category_path, new_filename)
                        page.save(save_path, "JPEG")

                    # Deletes original PDF and keeps only converted images
                    os.remove(pdf_path)
                    print(f"  Converted & Split: {filename} ({len(pages)} pages)")

if __name__ == "__main__":
    split_and_convert_pdfs(DATASET_PATH)


