import os
import re
import random
import numpy as np
from PIL import Image, ImageEnhance

class PneumoniaDatasetLoader:
    """
    Multimodal Dataset Loader for Chest X-Ray & CT Scan Pneumonia Detection.
    Extracts Subject IDs (e.g. 'person19' for X-Ray, 'ct_patient12' for CT Scans)
    to enable strict Cross-Subject Validation (GroupKFold) with zero patient data leakage.
    """
    def __init__(self, img_size=(150, 150)):
        self.img_size = img_size

    @staticmethod
    def extract_subject_id(filename, modality='xray'):
        """
        Parses patient/subject ID from filename.
        e.g., 'person19_bacteria_62.jpeg' -> 'person19'
              'ct_patient12_pneumonia_4.jpeg' -> 'ct_patient12'
              'NORMAL2-IM-0381-0001.jpeg' -> 'IM-0381'
        """
        base = os.path.basename(filename)
        
        # Match ct_patient pattern
        match_ct = re.search(r'(ct_patient\d+|patient\d+)', base, re.IGNORECASE)
        if match_ct:
            return match_ct.group(1).lower()

        # Match person pattern like person123
        match_person = re.search(r'(person\d+)', base, re.IGNORECASE)
        if match_person:
            return match_person.group(1).lower()
        
        # Match IM pattern
        match_im = re.search(r'(IM-\d+)', base)
        if match_im:
            return match_im.group(1)

        # Fallback hash
        prefix = 'ct_subj' if modality == 'ct_scan' else 'xray_subj'
        clean_name = re.sub(r'[\d_\.\-\s]+', '', base)
        return f"{prefix}_{hash(clean_name) % 1000:03d}"

    def generate_synthetic_dataset(self, base_dir, modality='xray', num_subjects=50, images_per_subject=6):
        """
        Generates synthetic Chest X-Ray or CT Scan multi-patient image dataset
        if local raw Kaggle dataset is not present in data_dir.
        """
        mod_dir = os.path.join(base_dir, modality)
        os.makedirs(os.path.join(mod_dir, 'NORMAL'), exist_ok=True)
        os.makedirs(os.path.join(mod_dir, 'PNEUMONIA'), exist_ok=True)

        records = []
        seed = 42 if modality == 'xray' else 99
        np.random.seed(seed)
        random.seed(seed)

        prefix = 'person' if modality == 'xray' else 'ct_patient'

        for i in range(1, num_subjects + 1):
            subject_id = f"{prefix}{i}"
            is_pneumonia = i % 3 != 0 # 66% pneumonia, 33% normal
            category = 'PNEUMONIA' if is_pneumonia else 'NORMAL'
            
            base_intensity = np.random.randint(90, 140) if modality == 'xray' else np.random.randint(40, 80)
            opacity_scale = 0.85 if is_pneumonia else 0.40

            for img_idx in range(1, images_per_subject + 1):
                fname = f"{subject_id}_{category.lower()}_{img_idx}.jpeg"
                fpath = os.path.join(mod_dir, category, fname)

                if modality == 'ct_scan':
                    img_arr = self._create_synthetic_ct_scan(base_intensity, opacity_scale, is_pneumonia)
                else:
                    img_arr = self._create_synthetic_xray(base_intensity, opacity_scale, is_pneumonia)

                img = Image.fromarray(img_arr)
                img.save(fpath, format='JPEG', quality=90)

                records.append({
                    'filepath': fpath,
                    'filename': fname,
                    'label': 1 if is_pneumonia else 0,
                    'category': category,
                    'subject_id': subject_id,
                    'modality': modality
                })

        print(f"[DatasetLoader] Synthesized {len(records)} {modality.upper()} images across {num_subjects} subjects in {mod_dir}")
        return records

    def _create_synthetic_xray(self, base_intensity, lung_opacity, is_pneumonia):
        """Creates synthetic projection Chest X-Ray matrix."""
        H, W = self.img_size
        img = np.full((H, W), base_intensity, dtype=np.float32)

        y, x = np.ogrid[:H, :W]
        center_x = W / 2.0
        spine = np.exp(-((x - center_x) ** 2) / 30.0) * 50.0
        img += spine

        left_lung = np.exp(-(((x - W*0.3)**2)/(W*2.5) + ((y - H*0.5)**2)/(H*3.5))) * 70.0
        right_lung = np.exp(-(((x - W*0.7)**2)/(W*2.5) + ((y - H*0.5)**2)/(H*3.5))) * 70.0
        img -= (left_lung + right_lung)

        if is_pneumonia:
            patch1 = np.exp(-(((x - W*0.35)**2)/450.0 + ((y - H*0.55)**2)/450.0)) * (100.0 * lung_opacity)
            patch2 = np.exp(-(((x - W*0.65)**2)/350.0 + ((y - H*0.45)**2)/350.0)) * (90.0 * lung_opacity)
            img += (patch1 + patch2)

        noise = np.random.normal(0, 4.0, (H, W))
        return np.clip(img + noise, 0, 255).astype(np.uint8)

    def _create_synthetic_ct_scan(self, base_intensity, opacity_scale, is_pneumonia):
        """Creates synthetic axial cross-sectional CT Scan slice matrix."""
        H, W = self.img_size
        y, x = np.ogrid[:H, :W]
        cx, cy = W / 2.0, H / 2.0

        # Thoracic body contour (bright subcutaneous tissue & ribs)
        dist = np.sqrt((x - cx)**2 + (y - cy)**2)
        body = (dist < (W * 0.44)).astype(np.float32) * 160.0

        # Low density lung parenchyma (dark air regions)
        left_lung_ct = (((x - W*0.35)**2)/(W*1.8)**2 + ((y - cy)**2)/(H*2.2)**2) < 0.08
        right_lung_ct = (((x - W*0.65)**2)/(W*1.8)**2 + ((y - cy)**2)/(H*2.2)**2) < 0.08

        img = body
        img[left_lung_ct | right_lung_ct] = 25.0 # Low attenuation lung air

        # Ground-Glass Opacities (GGO) & Peripheral Consolidations in CT
        if is_pneumonia:
            ggo1 = np.exp(-(((x - W*0.32)**2)/220.0 + ((y - H*0.58)**2)/220.0)) * (140.0 * opacity_scale)
            ggo2 = np.exp(-(((x - W*0.68)**2)/180.0 + ((y - H*0.52)**2)/180.0)) * (130.0 * opacity_scale)
            img += (ggo1 + ggo2)

        # Subcutaneous fat background
        img[dist >= (W * 0.44)] = 10.0
        noise = np.random.normal(0, 3.0, (H, W))
        return np.clip(img + noise, 0, 255).astype(np.uint8)

    def load_dataset_index(self, data_dir, modality='xray'):
        """Scans dataset directory for NORMAL and PNEUMONIA folders for specified modality."""
        records = []
        mod_dir = os.path.join(data_dir, modality) if os.path.exists(os.path.join(data_dir, modality)) else data_dir

        for category in ['NORMAL', 'PNEUMONIA']:
            cat_dir = os.path.join(mod_dir, category)
            if not os.path.exists(cat_dir):
                continue
            
            for root, _, files in os.walk(cat_dir):
                for f in files:
                    if f.lower().endswith(('.png', '.jpg', '.jpeg', '.dcm')):
                        fpath = os.path.join(root, f)
                        subj = self.extract_subject_id(f, modality=modality)
                        records.append({
                            'filepath': fpath,
                            'filename': f,
                            'label': 1 if category == 'PNEUMONIA' else 0,
                            'category': category,
                            'subject_id': subj,
                            'modality': modality
                        })
        return records

    def preprocess_image(self, img_path, augment=False):
        """Loads image, resizes, normalizes to [0, 1] for model input."""
        try:
            img = Image.open(img_path).convert('L')
        except Exception:
            img = Image.new('L', self.img_size, color=64)

        img = img.resize(self.img_size, Image.Resampling.BILINEAR)

        if augment:
            if random.random() > 0.5:
                img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            angle = random.uniform(-8, 8)
            img = img.rotate(angle)
            enhancer = ImageEnhance.Brightness(img)
            img = enhancer.enhance(random.uniform(0.88, 1.12))

        arr = np.array(img, dtype=np.float32) / 255.0
        arr = np.expand_dims(arr, axis=0) # (1, H, W)
        return arr
