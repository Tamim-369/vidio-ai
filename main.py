from src.utils.file_helpers import ensure_dirs, cleanup_temp
from src.services.data_source import research
from src.services.script_builder import build_script
from src.services.asset_fetcher import fetch_assets
from src.services.tts import generate_audio
from src.services.video_assembler import assemble


def run(topic: str):
    ensure_dirs()

    print(f"\n🔍 Researching: {topic}")
    raw_data = research(topic)

    print(f"\n📝 Building script...")
    script = build_script(topic, raw_data)
    print(f"   {len(script['lines'])} lines generated")
    with open("./script.json", "w") as f:
        import json
        json.dump(script, f, indent=2)
    print(f"\n🖼  Fetching images...")
    script["lines"] = fetch_assets(script["lines"], topic=topic)

    print(f"\n🎙  Generating voiceover...")
    script["lines"] = generate_audio(script["lines"])

    print(f"\n🎬 Assembling video...")
    output = assemble(script)

    cleanup_temp()
    print(f"\n✅ Done! Video saved to: {output}\n")


if __name__ == "__main__":
    run("Why Are you wasting time on social media?")
