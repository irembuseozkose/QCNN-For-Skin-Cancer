from pathlib import Path
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.qcnn.preprocess import PreprocessConfig, run_preprocessing


if __name__ == "__main__":
    cfg = PreprocessConfig(
        data_dir=Path("data/raw/Train"),
        output_dir=Path("data/processed/features_q"),

        image_size=(16, 16),
        color_mode="grayscale",

        test_size=0.15,
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

    print("Preprocessing tamamlandı.")
    print("Kaydedilen klasör:", cfg.output_dir)
    print("Train shape:", metadata["x_train_shape"])
    print("Val shape:", metadata["x_val_shape"])
    print("Test shape:", metadata["x_test_shape"])