"""Glassbox explorer: a Gradio interface over glassbox.api.

Run from the repo root:  python ui/app.py          (real model, needs the Hugging Face login in README)
                         python ui/app.py --fake   (sample data, no download or GPU)
"""
import argparse
from pathlib import Path

import gradio as gr

from glassbox.api import STEERING_LIMIT, Glassbox, load_tour
from glassbox.fake import FakeGlassbox

TOURS_DIR = Path(__file__).resolve().parent.parent / "tours"
STEERING_ROWS = 3
STEERING_STEP = 50
MAX_NEW_TOKENS = 60
DIFF_TOP_N = 20
FEATURE_HEADERS = ["Feature", "Label", "Activation"]
DIFF_HEADERS = ["Feature", "Label", "Mean A", "Mean B", "Change (B - A)"]


def steering_from_inputs(feature_ids, strengths) -> dict[int, float]:
    """Pair each feature-ID box with its slider. Empty IDs and zero strengths are skipped."""
    steering: dict[int, float] = {}
    for feature_id, strength in zip(feature_ids, strengths):
        if feature_id is None or not strength:
            continue
        feature_id = int(feature_id)
        if feature_id in steering:
            # Adding the two rows could silently exceed STEERING_LIMIT, so refuse instead.
            raise ValueError(f"Feature {feature_id} is in more than one steering row.")
        steering[feature_id] = float(strength)
    return steering


def reply_tokens(result) -> list:
    return [t for t in result.tokens if t.is_generated]


def token_spans(gb, result) -> list[tuple[str, str | None]]:
    """Reply tokens for gr.HighlightedText, coloured by each token's strongest feature."""
    return [(t.token, gb.label(t.features[0][0]) if t.features else None) for t in reply_tokens(result)]


def feature_rows(gb, token) -> list[list]:
    return [[feature_id, gb.label(feature_id), round(activation, 1)] for feature_id, activation in token.features]


def diff_rows(diffs) -> list[list]:
    return [[d.feature_id, d.label, round(d.mean_a, 1), round(d.mean_b, 1), round(d.delta, 1)] for d in diffs]


def load_tours(tours_dir: Path) -> dict[str, dict]:
    """Tours keyed by title, in file-name order. A missing folder just means no tours."""
    tours = [load_tour(path) for path in sorted(Path(tours_dir).glob("*.json"))]
    return {tour["title"]: tour for tour in tours}


def tour_inputs(tour: dict) -> list:
    """Values for [prompt, max_new_tokens, *feature_ids, *strengths]; unused steering rows are cleared."""
    steering = list(tour["steering"].items())[:STEERING_ROWS]
    padding = [None] * (STEERING_ROWS - len(steering))
    feature_ids = [feature_id for feature_id, _ in steering] + padding
    strengths = [strength for _, strength in steering] + [0] * len(padding)
    return [tour["prompt"], tour.get("max_new_tokens", MAX_NEW_TOKENS), *feature_ids, *strengths]


def build_app(gb, fake: bool = False, tours_dir: Path = TOURS_DIR) -> gr.Blocks:
    tours = load_tours(tours_dir)
    # Decoding is greedy, so the unsteered reply for a prompt never changes and is safe to reuse.
    baseline_cache: dict[tuple[str, int], object] = {}

    def run(prompt, max_new_tokens, *steer_inputs):
        if not prompt or not prompt.strip():
            raise gr.Error("Type a prompt first.")
        try:
            steering = steering_from_inputs(steer_inputs[:STEERING_ROWS], steer_inputs[STEERING_ROWS:])
        except ValueError as err:
            raise gr.Error(str(err))
        max_new_tokens = int(max_new_tokens)
        key = (prompt, max_new_tokens)
        try:
            if key not in baseline_cache:
                baseline_cache[key] = gb.generate(prompt, max_new_tokens=max_new_tokens)
            baseline = baseline_cache[key]
            result = gb.generate(prompt, max_new_tokens=max_new_tokens, steering=steering) if steering else baseline
        except ValueError as err:  # e.g. a feature ID outside the SAE's range
            raise gr.Error(str(err))
        return (
            gr.Textbox(value=result.text, label="Steered reply" if steering else "Reply"),
            gr.Textbox(value=baseline.text, visible=bool(steering)),
            token_spans(gb, result),
            result,
            "",
            [],
        )

    def show_token(result, evt: gr.SelectData):
        if result is None:
            return "", []
        token = reply_tokens(result)[evt.index]
        description = f"{token.token!r} (position {token.position})"
        if not token.features:
            description += ": no features, because the model never reads its own final token"
        return description, feature_rows(gb, token)

    def pick_tour(title):
        if not title:
            return ["", *[gr.skip()] * (2 + 2 * STEERING_ROWS)]
        tour = tours[title]
        return [tour.get("description", ""), *tour_inputs(tour)]

    def run_diff(prompt_a, prompt_b):
        if not prompt_a.strip() or not prompt_b.strip():
            raise gr.Error("Type both prompts first.")
        return diff_rows(gb.diff(prompt_a, prompt_b, top_n=DIFF_TOP_N))

    with gr.Blocks(title="Glassbox") as app:
        gr.Markdown("# Glassbox\nSee which sparse-autoencoder features fire as Gemma 3 writes, and steer them.")
        if fake:
            gr.Markdown("**Fake mode:** sample data, not real model output.")

        with gr.Tab("Explore"):
            tour_picker = gr.Dropdown(
                choices=list(tours), value=None, label="Guided tour (fills in a prompt and steering)", visible=bool(tours)
            )
            # Tour descriptions come from tours/*.json in this repo, not from the user, so Markdown is safe here.
            tour_description = gr.Markdown()
            prompt = gr.Textbox(label="Prompt", lines=2)
            with gr.Accordion("Steering", open=False):
                gr.Markdown(
                    "Enter a feature ID and push it up or down. Strength is in feature-activation units: "
                    "around 400 changes the wording; much higher can break the reply."
                )
                max_new_tokens = gr.Slider(minimum=10, maximum=120, value=MAX_NEW_TOKENS, step=10, label="Max new tokens")
                feature_ids, strengths = [], []
                for row in range(STEERING_ROWS):
                    with gr.Row():
                        feature_ids.append(gr.Number(label=f"Feature ID {row + 1}", precision=0, minimum=0))
                        strengths.append(
                            gr.Slider(
                                minimum=-STEERING_LIMIT,
                                maximum=STEERING_LIMIT,
                                value=0,
                                step=STEERING_STEP,
                                label="Strength (activation units)",
                            )
                        )
            generate_button = gr.Button("Generate", variant="primary")
            with gr.Row():
                reply = gr.Textbox(label="Reply", interactive=False)
                baseline_reply = gr.Textbox(label="Unsteered reply", interactive=False, visible=False)
            tokens = gr.HighlightedText(
                label="Reply tokens (colour = strongest feature). Click a token to see its top features.",
                combine_adjacent=False,
                show_legend=False,
            )
            selected = gr.Textbox(label="Selected token", interactive=False)
            features = gr.Dataframe(headers=FEATURE_HEADERS, interactive=False)
            result_state = gr.State(None)

        with gr.Tab("Diff"):
            gr.Markdown("Compare two prompts: which features are most active in one but not the other.")
            with gr.Row():
                prompt_a = gr.Textbox(label="Prompt A", lines=2)
                prompt_b = gr.Textbox(label="Prompt B", lines=2)
            diff_button = gr.Button("Compare", variant="primary")
            diff_table = gr.Dataframe(headers=DIFF_HEADERS, interactive=False)

        # Generate and Diff share one model and one feature hook, so they must never run at the same time.
        tour_picker.change(
            pick_tour,
            inputs=tour_picker,
            outputs=[tour_description, prompt, max_new_tokens, *feature_ids, *strengths],
        )
        generate_button.click(
            run,
            inputs=[prompt, max_new_tokens, *feature_ids, *strengths],
            outputs=[reply, baseline_reply, tokens, result_state, selected, features],
            concurrency_id="model",
        )
        tokens.select(show_token, inputs=result_state, outputs=[selected, features])
        diff_button.click(run_diff, inputs=[prompt_a, prompt_b], outputs=diff_table, concurrency_id="model")
    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Glassbox explorer")
    parser.add_argument("--fake", action="store_true", help="use sample data: no model download, login or GPU")
    args = parser.parse_args()
    gb = FakeGlassbox.load() if args.fake else Glassbox.load()
    build_app(gb, fake=args.fake).launch()


if __name__ == "__main__":
    main()
