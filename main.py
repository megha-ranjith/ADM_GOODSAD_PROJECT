import os
import torch
import numpy as np
from torchvision import datasets, transforms, models
from PIL import Image
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

CATEGORIES = [
    'drink_bottle',
    'drink_can',
    'food_bottle',
    'food_box',
    'food_package',
    'cigarette_box'
]

IMG_SIZE = 224
DATA_ROOT = './GoodsAD'
BATCH_SIZE = 32      
MAX_TRAIN = 120       # Limit for faster run (change as needed)
MAX_TEST_GOOD = 60    # Limit good test images (for quick plots/AUROC)
MAX_TEST_ANOM = 60    # Limit defect images (per category)
HIST_FOLDER = 'images'
os.makedirs(HIST_FOLDER, exist_ok=True)

train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def extract_features_imagepaths(image_paths, transform, feature_extractor, batch_size=BATCH_SIZE):
    all_feats = []
    for i in range(0, len(image_paths), batch_size):
        batch_files = image_paths[i:i+batch_size]
        imgs = []
        for fname in batch_files:
            img = Image.open(fname).convert('RGB')
            imgs.append(transform(img))
        if not imgs:
            continue
        tensor_batch = torch.stack(imgs)
        with torch.no_grad():
            feats = feature_extractor(tensor_batch).view(len(tensor_batch), -1).cpu().numpy()
        all_feats.append(feats)
    if all_feats:
        return np.concatenate(all_feats, axis=0)
    else:
        return np.zeros((0, 512))

for CATEGORY in CATEGORIES:
    print(f"\n\n==== Analyzing category: {CATEGORY} ====")
    category_path = os.path.join(DATA_ROOT, CATEGORY)
    if not os.path.isdir(category_path):
        print(f"!! Category missing: {CATEGORY}, skipping.")
        continue

    # Model, feature extractor
    resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    resnet.eval()
    feature_extractor = torch.nn.Sequential(*list(resnet.children())[:-1])

    # TRAIN: Good images
    train_good_folder = os.path.join(category_path, 'train', 'good')
    train_good_files = [os.path.join(train_good_folder, f)
                        for f in os.listdir(train_good_folder)
                        if f.lower().endswith(('.jpg', '.png'))][:MAX_TRAIN]
    if not train_good_files:
        print(f"No train good images for {CATEGORY}. Skipping.")
        continue
    train_features = extract_features_imagepaths(train_good_files, train_transform, feature_extractor)

    # Memory bank (KNN)
    knn = NearestNeighbors(n_neighbors=1)
    knn.fit(train_features)

    # TEST: Good images
    test_good_folder = os.path.join(category_path, 'test', 'good')
    test_good_files = [os.path.join(test_good_folder, f)
                        for f in os.listdir(test_good_folder)
                        if f.lower().endswith(('.jpg', '.png'))][:MAX_TEST_GOOD]
    test_good_feats = extract_features_imagepaths(test_good_files, train_transform, feature_extractor)
    if len(test_good_feats) == 0:
        print(f"No 'good' images in {test_good_folder}\n")
        continue
    good_scores, _ = knn.kneighbors(test_good_feats)
    print(f"Test/good - Images: {len(test_good_files)} | Mean anomaly score: {good_scores.mean():.3f}")

    # TEST: Anomaly images (all folders except "good")
    TEST_ROOT = os.path.join(category_path, 'test')
    anomaly_folders = [d for d in os.listdir(TEST_ROOT)
                       if os.path.isdir(os.path.join(TEST_ROOT, d)) and d.lower() != 'good']
    anomaly_files = []
    for a_folder in anomaly_folders:
        folder_path = os.path.join(TEST_ROOT, a_folder)
        files = [os.path.join(folder_path, f)
                 for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.png'))]
        anomaly_files += files
    anomaly_files = anomaly_files[:MAX_TEST_ANOM]
    if anomaly_files:
        anomaly_feats = extract_features_imagepaths(anomaly_files, train_transform, feature_extractor)
        anomaly_scores, _ = knn.kneighbors(anomaly_feats)
        print(f"Test/anomaly - Images: {len(anomaly_files)} | Mean anomaly score: {anomaly_scores.mean():.3f}")
    else:
        print("No anomaly images found.")
        continue

    # Save per-category histogram
    plt.figure(figsize=(8,5))
    plt.hist(good_scores.flatten(), bins=30, alpha=0.7, label='good', color='green')
    plt.hist(anomaly_scores.flatten(), bins=30, alpha=0.7, label='anomaly', color='red')
    plt.xlabel('Anomaly Score (KNN Distance)')
    plt.ylabel('Number of Images')
    plt.legend()
    plt.title(f'{CATEGORY}: Test Set Anomaly Scores')
    plt.tight_layout()
    plt.savefig(os.path.join(HIST_FOLDER, f"{CATEGORY}_hist.png"), dpi=300)
    plt.close()
    print(f"Saved {CATEGORY} histogram to images/{CATEGORY}_hist.png")

    # AUROC
    y_true = np.concatenate([np.zeros(len(good_scores)), np.ones(len(anomaly_scores))])
    y_scores = np.concatenate([good_scores.flatten(), anomaly_scores.flatten()])
    auroc = roc_auc_score(y_true, y_scores)
    print(f"AUROC: {auroc:.3f}")

    # Also save a side-by-side good/anomaly image
    try:
        img1 = Image.open(test_good_files[0]).convert('RGB')
        img2 = Image.open(anomaly_files[0]).convert('RGB')
        max_height = max(img1.height, img2.height)
        img1 = img1.resize((int(img1.width * max_height / img1.height), max_height))
        img2 = img2.resize((int(img2.width * max_height / img2.height), max_height))
        pair = Image.new('RGB', (img1.width + img2.width, max_height))
        pair.paste(img1, (0, 0))
        pair.paste(img2, (img1.width, 0))
        pairfile = os.path.join(HIST_FOLDER, f"{CATEGORY}_good_defect_pair.png")
        pair.save(pairfile)
        print(f"Saved example normal-defect pair: {pairfile}")
    except Exception as e:
        print(f"Could not create example pair for {CATEGORY}: {e}")


