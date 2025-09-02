import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_image():
    """Create a sample PIL Image for testing."""
    if not HAS_PIL or not HAS_NUMPY:
        pytest.skip("PIL or numpy not available")
    img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    return Image.fromarray(img_array)


@pytest.fixture
def sample_image_path(temp_dir, sample_image):
    """Create a sample image file and return its path."""
    if not HAS_PIL:
        pytest.skip("PIL not available")
    image_path = temp_dir / "test_image.jpg"
    sample_image.save(image_path)
    return image_path


@pytest.fixture
def sample_config():
    """Provide sample configuration dictionary."""
    return {
        "model_path": "models/yolov7.pt",
        "confidence": 0.25,
        "iou_threshold": 0.45,
        "image_size": 640,
        "device": "cpu"
    }


@pytest.fixture
def mock_model():
    """Mock YOLO model for testing."""
    mock = MagicMock()
    mock.names = {0: "person", 1: "bicycle", 2: "car"}
    mock.return_value = MagicMock()
    return mock


@pytest.fixture
def mock_detection_results():
    """Mock detection results for testing."""
    return [
        {
            "class": 0,
            "confidence": 0.85,
            "bbox": [100, 100, 200, 200],
            "label": "person"
        },
        {
            "class": 2,
            "confidence": 0.92,
            "bbox": [300, 150, 450, 350],
            "label": "car"
        }
    ]


@pytest.fixture
def mock_opencv_image():
    """Mock OpenCV image (numpy array)."""
    if not HAS_NUMPY:
        pytest.skip("numpy not available")
    return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)


@pytest.fixture
def empty_image_dir(temp_dir):
    """Create an empty directory for image testing."""
    img_dir = temp_dir / "images"
    img_dir.mkdir()
    return img_dir


@pytest.fixture
def sample_yaml_config(temp_dir):
    """Create a sample YAML configuration file."""
    config_path = temp_dir / "config.yaml"
    config_content = """
model:
  path: "models/best.pt"
  confidence: 0.25
  iou_threshold: 0.45

detection:
  image_size: 640
  device: "cpu"
  
classes:
  - person
  - bicycle  
  - car
  - motorcycle
"""
    config_path.write_text(config_content)
    return config_path


@pytest.fixture
def mock_torch_device():
    """Mock torch device for testing."""
    with patch("torch.cuda.is_available") as mock_cuda:
        mock_cuda.return_value = False
        yield "cpu"


@pytest.fixture
def mock_file_system(temp_dir):
    """Create a mock file system structure for testing."""
    # Create directory structure
    (temp_dir / "models").mkdir()
    (temp_dir / "images").mkdir()
    (temp_dir / "outputs").mkdir()
    
    # Create mock model file
    model_file = temp_dir / "models" / "test_model.pt"
    model_file.touch()
    
    return {
        "root": temp_dir,
        "models": temp_dir / "models",
        "images": temp_dir / "images", 
        "outputs": temp_dir / "outputs",
        "model_file": model_file
    }


@pytest.fixture
def mock_ui_components():
    """Mock UI components for GUI testing."""
    return {
        "main_window": MagicMock(),
        "image_label": MagicMock(),
        "progress_bar": MagicMock(),
        "status_label": MagicMock(),
        "detection_list": MagicMock()
    }


@pytest.fixture(autouse=True)
def clean_environment():
    """Clean up environment variables and settings before each test."""
    # Store original values
    original_env = dict(os.environ)
    
    yield
    
    # Restore environment
    os.environ.clear()
    os.environ.update(original_env)


@pytest.fixture
def capture_logs(caplog):
    """Capture log messages during testing."""
    return caplog


@pytest.fixture(scope="session")
def test_data_dir():
    """Directory containing test data files."""
    return Path(__file__).parent / "data"