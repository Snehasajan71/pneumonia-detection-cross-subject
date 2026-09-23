import os
import re
import random
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

class PneumoniaDatasetLoader:
    """
    Dataset loader for Chest X-Ray Pneumonia detection with Patient Subject Grouping.
    Extracts Subject IDs (e.g. 'person100' from 'person100_bacteria_189.jpeg') to enable
    strict Cross-Subject Validation (GroupKFold) with zero patient data leakage.
    """
    def __init__(self, img_size=(150, 150)):
        self.img_size = img_size

    @staticmethod
    def extract_subject_id(filename):
        """
        Parses patient/subject ID from filename.
        e.g., 'person19_bacteria_62.jpeg' -> 'person19'
              'person100_normal_1.jpeg' -> 'person100'
              'NORMAL2-IM-0381-0001.jpeg' -> 'IM-0381'
        """
        base = os.path.basename(filename)
        # Match person pattern like person123
        match = re.search(r'(person\d+)', base, re.IGNORECASE)
        if match:
            return match.group(1).lower()
        
        # Match IM pattern
        match_im = re.search(r'(IM-\d+)', base)
        if match_im:
            return match_im.group(1)

        # Fallback to hash prefix if no standard patient pattern
        clean_name = re.sub(r'[\d_\.\-\s]+', '', base)
        return f"subject_{hash(clean_name) % 1000:03d}"

    def generate_synthetic_dataset(self, base_dir, num_subjects=50, images_per_subject=6):
        """
        Generates a synthetic chest X-ray dataset structured by patient subject IDs
        if raw Kaggle dataset is not locally present.
        """
        os.makedirs(os.path.join(base_dir, 'NORMAL'), exist_ok=True)
        os.makedirs(os.path.join(base_dir, 'PNEUMONIA'), exist_ok=True)

        records = []
        np.random.seed(42)
        random.seed(42)

        for i in range(1, num_subjects + 1):
            subject_id = f"person{i}"
            # Assign label to subject (70% pneumonia, 30% normal to match dataset balance)
            is_pneumonia = i % 3 != 0  
            category = 'PNEUMONIA' if is_pneumonia else 'NORMAL'
            
            # Subject specific baseline intensity & texture characteristics
            base_intensity = np.random.randint(90, 140)
            lung_opacity = 0.85 if is_pneumonia else 0.45

            for img_idx in range(1, images_per_subject + 1):
                fname = f"{subject_id}_{category.lower()}_{img_idx}.jpeg"
                fpath = os.path.join(base_dir, category, fname)

                # Create synthetic X-ray image (150x150) with synthetic lung structures & opacities
                img_arr = self._create_synthetic_xray(base_intensity, lung_opacity, is_pneumonia)
                img = Image.fromarray(img_arr)
                img.save(fpath, format='JPEG', quality=90)

                records.append({
                    'filepath': fpath,
                    'filename': fname,
                    'label': 1 if is_pneumonia else 0,
                    'category': category,
                    'subject_id': subject_id
                })

        print(f"[DatasetLoader] Generated synthetic dataset at {base_dir} with {len(records)} images across {num_subjects} subjects.")
        return records

    def _create_synthetic_xray(self, base_intensity, lung_opacity, is_pneumonia):
        """Creates realistic synthetic Chest X-Ray matrix."""
        H, W = self.img_size
        img = np.full((H, W), base_intensity, dtype=np.float32)

        # Rib cage & spine structure simulation
        y, x = np.ogrid[:H, :W]
        center_x = W / 2.0
        spine = np.exp(-((x - center_x) ** 2) / 30.0) * 50.0
        img += spine

        # Left & Right lung fields (darker low density regions)
        left_lung = np.exp(-(((x - W*0.3)**2)/(W*2.5) + ((y - H*0.5)**2)/(H*3.5))) * 70.0
        right_lung = np.exp(-(((x - W*0.7)**2)/(W*2.5) + ((y - H*0.5)**2)/(H*3.5))) * 70.0
        img -= (left_lung + right_lung)

        # Pneumonia opacity infiltrates (white patchy consolidations)
        if is_pneumonia:
            opacity_patch1 = np.exp(-(((x - W*0.35)**2)/450.0 + ((y - H*0.55)**2)/450.0)) * (100.0 * lung_opacity)
            opacity_patch2 = np.exp(-(((x - W*0.65)**2)/350.0 + ((y - H*0.45)**2)/350.0)) * (90.0 * lung_opacity)
            img += (opacity_patch1 + opacity_patch2)

        # Noise & smoothing
        noise = np.random.normal(0, 5.0, (H, W))
        img = np.clip(img + noise, 0, 255).astype(np.uint8)
        return img

    def load_dataset_index(self, data_dir):
        """
        Scans data_dir for NORMAL and PNEUMONIA folders, parses subject IDs.
        """
        records = []
        for category in ['NORMAL', 'PNEUMONIA']:
            cat_dir = os.path.join(data_dir, category)
            if not os.path.exists(cat_dir):
                continue
            
            for root, _, files in os.walk(cat_dir):
                for f in files:
                    if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                        fpath = os.path.join(root, f)
                        subj = self.extract_subject_id(f)
                        records.append({
                            'filepath': fpath,
                            'filename': f,
                            'label': 1 if category == 'PNEUMONIA' else 0,
                            'category': category,
                            'subject_id': subj
                        })
        return records

    def preprocess_image(self, img_path, augment=False):
        """
        Loads image, resizes to target size, applies optional data augmentation,
        and normalizes pixel values to [0, 1].
        """
        try:
            img = Image.open(img_path).convert('L') # Convert to grayscale
        except Exception:
            # Fallback black image if file unreadable
            img = Image.new('L', self.img_size, color=128)

        img = img.resize(self.img_size, Image.Resampling.BILINEAR)

        if augment:
            # Random horizontal flip
            if random.random() > 0.5:
                img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            # Random rotation (-10 to +10 degrees)
            angle = random.uniform(-10, 10)
            img = img.rotate(angle)
            # Random brightness shift
            enhancer = ImageEnhance.Brightness(img)
            img = enhancer.enhance(random.uniform(0.85, 1.15))

        arr = np.array(img, dtype=np.float32) / 255.0
        # Expand channel dimension (1, H, W) or (H, W, 1)
        arr = np.expand_dims(arr, axis=0) # (1, H, W) for PyTorch format
        return arr
