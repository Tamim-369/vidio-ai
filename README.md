# videoai_quote — parody quote shorts

A batch pipeline that turns AI-written jokes into finished, uploaded YouTube
Shorts. Each video is one of three layouts, and a batch cycles through all of
them.

```bash
uv run src/main.py --batch=10 --no-upload
```

Roughly 45 seconds per video. Most of that is neural TTS on CPU, not the
language model.

---

## What it does

```
character → quote (Groq) → voiceover (Chatterbox) → quote card → music → upload
```

1. **Plan the video.** A batch decides every slot's layout and cast up front, so
   the schedule is known before a single token is spent. Both come from the
   video's number: the lead is the voice at `(N - 1) % len(enabled)` and the
   layout is the nth entry of the layout cycle, so the two run out of phase. The
   number is the only state — a batch that resumes picks up exactly where the
   last one stopped, with no separate cursor to fall out of step.
2. **Write the quotes.** Each character decides its own subject — Don Tzu writes
   military strategy, Brolexander writes gym discipline — so the joke and the
   narrator always agree. One call per *character*, however many quotes it says.
3. **Speak them.** Chatterbox clones the voice from a short reference clip, so
   each personality is audibly itself. Lines are grouped by character, so each
   voice's conditionals load once no matter how many lines that character has.
4. **Render a card per quote.** Each quote over that character's photo, cut to
   the length of the audio it was spoken in, then concatenated.
5. **Mix, title, publish.** Background music, a numbered title and description,
   and an upload — or keep it local with `--no-upload`.

## The three layouts

| type | who speaks | quotes | shape |
| ---- | ---------- | ------ | ----- |
| 1 | 1 character | 3 | a monologue |
| 2 | 3 characters | 1 each | a round table |
| 3 | 2 characters | 1 each | a quick two-hander |

Types 2 and 3 are the same rule at different widths. Only a monologue repeats
one character, because it is the only layout that needs to.

Over ten videos the layouts run `1,2,3,1,3,2,1,2,3,1` while the lead runs
`Trump → Tate → Arnold` and repeats — 4 monologues, 3 round tables, 3
two-handers, and 27 quotes from 19 model calls. The two cycles are deliberately
out of phase: in phase they would repeat every three videos and the schedule
would be predictable from the first two.

## Numbering

Titles and descriptions carry `| Wordz Of Wizdom #N`, so the channel reads as a
numbered library. The counter is persisted in `video_number.json` and **advances
when a video completes, not when it is published** — so a `--no-upload` preview
consumes numbers. A gap is invisible to a viewer; two videos sharing a number is
not. The number is also the plan's phase, which is why two runs in a row do not
replay the same opening layouts, and why a video that failed keeps its slot.

## Why the layouts vary

A batch used to ship one character saying one quote, on the argument that a
second line only adds narration and a second still card. That was true of the
*card* and false of the *video*, which is a sequence of them. Ten identical
monologues are not variety, and changing the narrator while the shape stays
fixed is the least interesting way to vary a feed. So the shape is planned per
video, and a monologue is the minority layout rather than all of them.

## Setup

```bash
uv sync
cp .env.example .env      # then fill in the keys
```

Dependencies are declared in `pyproject.toml` and pinned to the exact versions
the voices were tuned against. There is no `requirements.txt`; `uv sync` from
the lock is the only supported install.

| Variable | Required | Why |
| --- | --- | --- |
| `GROQ_API_KEY` | yes | Quote generation. |
| `GROQ_API_KEY_SECOND`, `GROQ_API_KEY_THIRD` | no | The client rotates to these on a rate limit, which happens on batches over about ten videos. |
| `YOUTUBE_CLIENT_SECRETS` | only for `--upload` | Path to the OAuth client secrets JSON. Defaults to `client_secrets.json` in the project root. |
| `AUTO_PUBLISH` | no | `1` publishes by default; `--upload`/`--no-upload` override it. Off by default. |
| `HF_HOME` | no | Where model weights are cached. |
| `GROQ_QUOTE_MODEL` | no | Defaults to `openai/gpt-oss-20b`. |

Speech synthesis runs on CPU and needs no key. It does need the Chatterbox
weights, about 3.3GB, which land in `HF_HOME` on first run.

YouTube uses the OAuth installed-app flow: the first `--upload` prints a URL,
you paste it back, and the resulting `token.json` is reused. Both that and
`client_secrets.json` are gitignored.

## Commands

```bash
uv run src/main.py                              # one video
uv run src/main.py --batch=10 --no-upload       # ten, kept local
uv run src/main.py --batch=5 --script-only      # just the quotes, no render
uv run src/main.py --voice=donald-trump         # force the lead of every video
uv run src/main.py --batch=3 --upload           # publish to YouTube
```

Five flags, all of them. There is no format, seed, or line-count knob, because
there is nothing left to configure.

## Layout

```
src/
  main.py                    CLI: four flags
  agents/
    quotes/                  joke generation + validation + dedupe
    voice_cast/              the character registry (who, subject, voice, face)
    voiceover/               Chatterbox TTS: chunking, workers, DSP, timing
    visuals/                 the quote card renderer
    soundtrack/              background music
    publish/                 YouTube upload + deterministic metadata
    video/                   the batch planner, pipeline loop and title rules
    completion/              the LLM client, with key rotation
  assets/voice_refs/         reference clips the voices are cloned from
  faces/                     the portrait on each card
```

Each agent owns one job and is callable on its own. `agents/video/pipeline.py`
is the only place that knows the order.

## Design notes

**Quotes are validated, not trusted.** A candidate is rejected for being
meta-commentary, for being multi-line, for falling outside the length window its
character's prompt asks for, for reading like a fact rather than a joke, or for
duplicating anything already used. The prompt treats sincere wisdom as a
failure, so the filter is deliberately strict.

**No quote is ever used twice.** Every accepted quote is appended to
`used_quotes.json` and checked against on the way in. The file is local mutable
state, deliberately untracked.

**The text client has somewhere else to go.** A batch of ten is nineteen calls
against a rate limit, and a hard 429 mid-batch means a half-finished run. The
client falls back Groq → Gemini → Cloudflare, rotates through three Groq keys
first, and only gives up once all three tiers are exhausted. Everything after
the first tier is optional insurance, so the required setup stays at one key.

**A video is retried before it is given up on.** Neural TTS is the expensive step
and a single call can fail on its own — an unlucky quote, a transient model
error, memory pressure. A video is several quotes now, so a failure partway
through would otherwise cost the whole slot. A batch retries each video twice,
re-running the whole build, which also draws a fresh set of quotes; generation
deduplicates against everything already used, so a retry cannot repeat the ones
that failed. Only a video that fails every attempt is reported and skipped.

**Metadata is computed, not generated.** An LLM asked for ten descriptions
writes ten near-identical paragraphs, and the only cure is watching for
collisions and regenerating. Building the description from the video's own
quotes makes a duplicate structurally impossible and takes a step off the batch.

**Titles are built from the quote.** `<the quote> | Wordz Of Wizdom #N`, capped
at 100 characters on a word boundary because YouTube rejects longer titles
outright. The quote is fitted to the room the suffix leaves *before* the suffix
is added, so the number — the part a viewer uses to find the video — is never
what gets cut. The pseudonym is deliberately absent: a real quote runs 58–98
characters, and prefixing a name would clip the hook to about 63. The speaker is
named in the description instead, where there is room. Since quotes are
deduplicated, titles are unique for free.

**The description is quote-led.** The first quote leads unnumbered, because it
is also the title and numbering it `1.` would imply a list it is not the first
item of. The rest are numbered and attributed to the pseudonym that spoke them,
which is what lets a viewer tell a round table apart at a glance. Then the
parody disclaimer, the number, and a hashtag per speaker.

**Parody, and labelled as such.** These are fabricated jokes written in the
style of public figures, credited to invented names. Every description carries
a disclaimer, and Chatterbox's audio watermark is explicitly stripped — the
model ships one that marks output as machine-generated, and that is the correct
behaviour for this content, so it stays off.

## Verifying a change

There is no test suite. Changes are checked by running the thing:

```bash
uv run --with pyflakes python -m pyflakes src/   # unused/undefined names
uv run src/main.py --batch=10 --script-only      # the plan, titles, numbering
uv run src/main.py --batch=1 --no-upload         # one full render, kept local
```

`--script-only` is the cheap one and the one to reach for: it exercises
planning, quote generation, validation, dedupe, the title and description
builders and the number counter, and stops before the 45-second render.

Two things it will not catch, and which cost real time when they slip:

- **A title over 100 characters.** YouTube rejects the whole upload at the API,
  after a full render. `build_numbered_title` fits the quote to whatever the
  suffix leaves, and a number reaching five digits makes the suffix longer.
- **A quote that repeats.** The dedupe pool is what stops it, and it lives in
  `used_quotes.json` next to this file.
