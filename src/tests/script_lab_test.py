"""Standalone lab test for the staged script pipeline (does NOT touch main.py).

Run:
    uv run python -u src/tests/script_lab_test.py [--theme arnold|trump] [--topic "..."] [--story "..."]

Prints every layer's output so we can eyeball script quality and virality
before this logic ever moves into the main pipeline. Default input is the
Ghost Army (a real, fact-dense WWII story) so extraction has plenty to grab.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.services.script_lab import run_pipeline

DEFAULT_TOPIC = "The Ghost Army"
DEFAULT_STORY = """The Ghost Army was a real United States Army unit that won battles without ever firing a shot. It was called the 23rd Headquarters Special Troops, and it used inflatable rubber tanks, fake radio chatter, and pretend generals to fool the Germans across Europe in World War Two. About 1,100 men served in it, most of them artists, engineers, and sound men rather than combat soldiers. They landed in Normandy in July 1944, and they kept their secret so well that the operation was classified until 1996. Their rubber tanks and artillery looked real from 500 yards, and the whole unit could pack up and move about 63 miles in a single night. They staged more than 20 deception operations in France, Luxembourg, and Germany. Their radio operators imitated entire phantom divisions, so German listeners heard more enemy units than actually existed - sometimes a force of 30,000 men. The sound trucks they used had speakers that could project the noise of tanks and infantry miles away. After the war, two members of this unit became famous: the fashion designer Bill Blass and the artist Ellsworth Kelly. They painted and inflated their way through a war, and their story stayed hidden for more than fifty years."""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--theme", default="arnold", choices=["arnold", "trump"])
    parser.add_argument("--topic", default=DEFAULT_TOPIC)
    parser.add_argument("--story", default=DEFAULT_STORY)
    args = parser.parse_args()

    result = run_pipeline(
        topic=args.topic,
        story=args.story,
        theme=args.theme,
    )

    print(f"\n{'='*70}\nRESULT\n{'='*70}")
    print(f"facts   : {len(result['facts'])}")
    print(f"lines   : {len(result['lines'])}")
    print(f"queries : {sum(len(l.get('queries', [])) for l in result['lines'])} "
          f"(>=2 per line expected)")


if __name__ == "__main__":
    main()