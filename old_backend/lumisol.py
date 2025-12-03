import typer
from enum import Enum
from scripts.pipeline.config import CONFIG
from scripts.pipeline.lidar_downloader import LidarDownloader
from scripts.pipeline.vector_filter import VectorFilter
from scripts.pipeline.data_generator import DataGenerator
from scripts.pipeline.model_trainer import ModelTrainer
from scripts.pipeline.visualizer import ModelVisualizer
from scripts.pipeline.post_processor import PostProcessor

app = typer.Typer(help="LumiSol Geospatial AI Pipeline")

# 1. Define Valid Architectures
class Architecture(str, Enum):
    mps = "mps"   # Apple Silicon
    cuda = "cuda" # NVIDIA GPU
    cpu = "cpu"   # Fallback

# 2. Define Optimal Configs per Architecture
ARCH_CONFIGS = {
    Architecture.mps: {
        "ACCELERATOR": "mps",
        "DEVICES": 1,
        "PRECISION": "32-true",      # Mixed precision is slow on M2
        "NUM_WORKERS": 0,            # macOS multiprocessing works best at 0
        "ACCUMULATE_BATCHES": 4      # Simulate larger batch on limited RAM
    },
    Architecture.cuda: {
        "ACCELERATOR": "gpu",
        "DEVICES": "auto",
        "PRECISION": "16-mixed",     # Tensor Cores go brrr 🚀
        "NUM_WORKERS": 4,            # Parallel loading ok on Linux/Windows
        "ACCUMULATE_BATCHES": 2      # Less accumulation needed if VRAM is high
    },
    Architecture.cpu: {
        "ACCELERATOR": "cpu",
        "DEVICES": "auto",
        "PRECISION": "32-true",
        "NUM_WORKERS": 0,
        "ACCUMULATE_BATCHES": 1
    }
}

@app.command()
def filter():
    """Step 1: Smart filtering of building vectors & tile selection."""
    typer.echo("📍 Step 1: Filtering Vectors...")
    VectorFilter(CONFIG).run()

@app.command()
def download():
    """Step 2: Download LiDAR tiles."""
    typer.echo("⬇️ Step 2: Downloading LiDAR Data...")
    LidarDownloader(CONFIG).run()

@app.command()
def generate():
    """Step 3: Generate AI-ready training chips."""
    typer.echo("🏭 Step 3: Generating Training Dataset...")
    DataGenerator(CONFIG).run()

@app.command()
def train(
    # New Flag: Architecture
    arch: Architecture = typer.Option(
        Architecture.mps, 
        "--arch", "-a", 
        help="Hardware architecture strategy (mps, cuda, cpu)"
    ),
    epochs: int = typer.Option(CONFIG["TRAIN_EPOCHS"], help="Override epochs"),
    batch_size: int = typer.Option(CONFIG["TRAIN_BATCH_SIZE"], help="Override batch size"),
    debug: bool = typer.Option(False, "--debug", "-d", help="Run fast sanity check on 1 batch")
):
    """Step 4: Train the Mask2Former model."""
    
    # 3. Merge Base Config with Architecture Overrides
    run_config = CONFIG.copy()
    arch_settings = ARCH_CONFIGS[arch]
    
    # Apply architecture specific settings
    run_config.update(arch_settings)
    
    # Apply CLI overrides
    run_config["TRAIN_EPOCHS"] = epochs
    run_config["TRAIN_BATCH_SIZE"] = batch_size
    run_config["DEBUG"] = debug
    
    typer.echo(f"🧠 Training on [{arch.value.upper()}] | Precision: {arch_settings['PRECISION']} | Workers: {arch_settings['NUM_WORKERS']}")
    
    ModelTrainer(run_config).run()

@app.command()
def test(
    samples: int = typer.Option(5, help="Number of images to generate")
):
    """
    Step 5: Visual Inspection. Runs the trained model on random samples.
    """
    typer.echo(f"📸 Running Visual Test on {samples} samples...")
    ModelVisualizer(CONFIG).run(num_samples=samples)

@app.command()
def post_process():
    """
    Step 6: Vectorize AI output, dilate by 0.5m, and regularize geometry.
    """
    typer.echo("📐 Step 6: Post-Processing & Regularization...")
    PostProcessor(CONFIG).run()

@app.command()
def all():
    """Run the ENTIRE pipeline (Filter -> Download -> Generate -> Train)."""
    filter()
    download()
    generate()
    train()

if __name__ == "__main__":
    app()
