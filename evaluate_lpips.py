import os

import lpips
from PIL import Image
from torchvision import transforms


def calculate_intra_lpips(folder1, device='cuda'):
    loss_fn = lpips.LPIPS(net='alex').to(device)
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )
    lpips_scores = []
    files1 = sorted(os.listdir(folder1))

    for file1 in files1:
        for file2 in files1:
            img1 = Image.open(os.path.join(folder1, file1)).convert('RGB')
            img2 = Image.open(os.path.join(folder1, file2)).convert('RGB')

            img1_tensor = transform(img1).unsqueeze(0).to(device)
            img2_tensor = transform(img2).unsqueeze(0).to(device)

            lpips_score = loss_fn(img1_tensor, img2_tensor).item()
            lpips_scores.append(lpips_score)

    return sum(lpips_scores) / len(lpips_scores)


def main(folder1):
    print("Calculating LPIPS...")
    intra_lpips_score = calculate_intra_lpips(folder1)
    print(f"Intra LPIPS Score: {intra_lpips_score}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Calculate LPIPS")
    parser.add_argument(
        "--folder1",
        type=str,
        help="Path to the first folder of images",
    )
    args = parser.parse_args()

    main(args.folder1)
