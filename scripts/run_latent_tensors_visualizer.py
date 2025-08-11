import argparse
from pathlib import Path
import sys

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))


from src.visualization.latent_tensors_visualizer import latent_tensors_visualizer

def get_prompt_groups(latents_dir, user_specified=None):
    if user_specified:
        return [g.strip() for g in user_specified.split(",")]
    else:
        return sorted([p.name for p in Path(latents_dir).iterdir() if p.is_dir()])

def main():
    parser = argparse.ArgumentParser(description="Run latent_tensors_visualizer on a latents directory.")
    parser.add_argument('--latents_dir', type=str, required=True, help='Path to the latents directory')
    parser.add_argument('--output_dir', type=str, required=True, help='Path to the output directory')
    parser.add_argument(
        "--prompt-groups", type=str, default=None,
        help="Comma-separated list of prompt group names (e.g., 'prompt_000,prompt_001'). If not passed, all subdirs in latents/ are used."
    )
    args = parser.parse_args()

    latents_dir = Path(args.latents_dir)
    output_dir = Path(args.output_dir)

    prompt_groups = get_prompt_groups(latents_dir, args.prompt_groups)

    latent_tensors_visualizer(
        group_tensors=None,
        latents_dir=latents_dir,
        output_dir=output_dir,
        prompt_groups=prompt_groups
    )

if __name__ == "__main__":
    main()
