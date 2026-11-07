- You are a Software Engineer agent.
- we are making a new project called `perception-voice`
  - we will compose this in our workspace root
    - do not edit any files under `tmp/`
  - use Python with `uv`
    - no file >500 lines
    - no function >50 lines
    - we have a gpu with CUDA support, prefer to use hardware acceleration when its an option
- we will take inspiration from a prior work; project `Whisper` located under `tmp/whisper/`
  - this is a project that is able to transcribe audio (from microphone input) to text (Speech-to-Text). offering two modes: AGENT (executes shell commands), and LISTEN (types out the words heard onto sendkey keyboard input)
  - read its `README.md` to understand the project
  - you can also use command `whisper --help` to understand its cli

problem statement:
- whisper is a good tool i have used a lot, but when multiple apps need to listen for voice commands, it requires each process to load a copy of the AI model into gpu vram, which doesn't scale well.

solution:
- we will decompose the solution this way:
  - take the part of whisper that successfully listens to audio and transcribes it to text
  - make it a process that can run as a systemd unit (always on)
  - it will remember all the words it hears in memory (for up to the last 30 minutes (configurable in `config.yml`))
    - until process is restarted; then it will forget
- this way now only one process is managing the whisper AI model in gpu vram
  - and we will make it accessible to other processes this way:
    - other processes will be able to communicate with this one (over unix domain socket located in workspace root (gitignored))
      - the cli for this process is basically able to operate in two modes:
        - `perception-voice serve`: start in server/service mode
        - `perception-voice client`: start in client mode (run, communicate with server, exit)
          - subcommands include:
            - `set <uid>`: set read marker to now
              - where `uid`: is any string, treated as a unique identifier (the read marker is stored per-uid)
            - `get <uid>`: get text since read marker, and update read marker to now
              - where `uid`: is any string, treated as a unique identifier (the read marker is stored per-uid)

example daily workflow:

- (computer starts up)
- (systemd launches perception-voice unit via `perception-voice serve`)

... later ...

- (a process starts up that needs to use voice; one called `mari`)
- (mari issues a command) `perception-voice client set mari`
- (perception-voice server sets read_marker[mari]=Date.now()`)

... after some time ...
- (perception-voice server hears audio (ie. from microphone) and translates it to text "turn on lights", storing it in memory with a timestamp)
... after some time ...

- (mari checks in) `perception-voice client get mari`
- (perception-voice server responds to perception-voice client with the text in buffer since last mari read marker was set): "turn on lights"
- (perception-voice client prints text to stdout and exits 0)
- (mari is able to use "turn on lights" to do something useful)



we are in PLAN MODE; that means you can only edit file `docs/PLAN.md` until I say EXIT PLAN MODE. my requests are therefore intended to iteratively improve upon our plan document.

restate the plan in the doc.
ask clarifying questions (toward bottom of the doc)

...

the audio feedback sfx is not necessary to keep.


the whisper model we're using
- is not committed to git
- is gitignored
- is one that i like; make sure we are using the same one as Whisper
- had a download script which was useful as part of first-time installation for new users
  - but just as long as on first startup
    - we are able to fetch the correct model 
      - that will be ideal



answers to open questions:

1. JSONL is preferable with both timestamp (incl. milliseconds and time zone) and utterance
2. each utterance is its own timestamped entry
3. get always advances the marker
4. configurable is good. it is located in workspace root normally.
5. you can decide exit codes. no new text should return empty string and exit 0.
6. no pidfile needed. no signal handling needed (other than to exit on SIGTERM as normal)
7. no need for auth
8. stdout for journald. provide a `-v` cli param to enable verbose logging. by default it will only output minimal to stdout, but with verbose logging it might log events (without necessarily logging the text it heard; maybe instead metadata like word count and time stamps)
9. speed is important but let's start with just the large model and i'll test it and let you know if we need more than that.
10. no need for health check.

---

- read `**/*.md` to understand this project.
- our plan has been implemented, tested, and confirmed working.

now we'll make our first use of this service.

it's unlikely i will use the former Whisper project again after this.

what we will do is add a new subcommand:

`perception-voice keyboard`

what this will do is just the keyboard part of whisper (copy that code pretty closely as it was known-good proven working)

it will run as a systemd service.
it will get its own perception-voice-keyboard.service (systemd unit file).

but when it starts up:
- it does not interact with gpu.
- it does not overlap with `perception-voice serve`
- it is essentially a specific/subset/implementation of `perception-voice client`

let me explain what i mean.

typical daily workflow:

- i wake up
- i turn on my pc
- perception-voice.service auto-starts
- i unmute my microphone
- perception-voice-keyboard.service auto-starts
- i place my cursor in a window (say, vscode) and ensure its editor textarea has focus
- i speak into my microphone "three blind mice"
- nothing happens. i realize i forgot to press Ctrl+Shift+Space hotkey which toggles on/off the keyboard's active listening mode.
  - i press the hotkey.
  - i hear the activation SFX play, indicating it is keyboard feature is now active.
- i speak again "hamsters are cute"
- instantly the editor textarea displays "hamsters are cute" (the keyboard input was sent)
- i press the hotkey.
  - i hear the deactivate SFX play. i feel confident knowing its no longer going to type when i click and do other  activities on my pc.
- later, i decide i want to use the dictation some more. i press hotkey, and speak "wouldn't it be nice?"
  - but immediately i catch myself and realize i changed my mind, i do not want the text to write to the keyboard-- so i hit the hotkey (when pressed before the Speech-to-text completes transcription, any text in the buffer is discarded instead of emitted to keyboard; see existing Whisper code to learn how this is implemented currently)
    - no text is emitted to keyboard (phew!)

