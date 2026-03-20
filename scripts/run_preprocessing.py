from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.qcnn.preprocess import PreprocessConfig, run_preprocessing


if __name__ == "__main__":
    cfg = PreprocessConfig(
        # FIX: train ve test klasörleri ayrı ayrı belirtiliyor
        train_dir=Path("data/raw/Train"),
        test_dir=Path("data/raw/Test"),
        output_dir=Path("data/processed/features_q_amplitude"),

        image_size=(16, 16),       # 16x16 grayscale = 256 = 2^8  ✓
        color_mode="grayscale",

        # val, train'den ayrılır — test'e dokunulmaz
        val_size=0.15,
        random_state=42,

        normalize_pixels=True,
        flatten=True,

        use_pca=False,
        n_components=None,

        encoding_mode="amplitude",
        angle_scale=3.141592653589793,

        save_intermediate_arrays=False,
    )

    metadata = run_preprocessing(cfg)

    print("\n" + "=" * 50)
    print("Preprocessing tamamlandı.")
    print("Kaydedilen klasör :", cfg.output_dir)
    print("Train shape        :", metadata["x_train_shape"])
    print("Val shape          :", metadata["x_val_shape"])
    print("Test shape         :", metadata["x_test_shape"])
    print("Sınıf sayısı       :", metadata["num_classes"])
    print("Label map          :", metadata["label_map"])
    print("=" * 50)