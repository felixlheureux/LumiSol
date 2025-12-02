import typer
from scripts.pipeline.config import CONFIG
from scripts.pipeline.lidar_downloader import LidarDownloader
from scripts.pipeline.vector_filter import VectorFilter
from scripts.pipeline.data_generator import DataGenerator
from scripts.pipeline.model_trainer import ModelTrainer

app = typer.Typer(help="LumiSol Geospatial AI Pipeline")

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
    epochs: int = typer.Option(CONFIG["TRAIN_EPOCHS"], help="Override epochs"),
    batch_size: int = typer.Option(CONFIG["TRAIN_BATCH_SIZE"], help="Override batch size")
):
    """Step 4: Train the Mask2Former model."""
    typer.echo(f"🧠 Step 4: Training Model ({epochs} epochs)...")
    
    # Allow CLI overrides of config
    run_config = CONFIG.copy()
    run_config["TRAIN_EPOCHS"] = epochs
    run_config["TRAIN_BATCH_SIZE"] = batch_size
    
    ModelTrainer(run_config).run()

@app.command()
def all():
    """Run the ENTIRE pipeline (Filter -> Download -> Generate -> Train)."""
    filter()
    download()
    generate()
    train()

if __name__ == "__main__":
    app()
