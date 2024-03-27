import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
import distinctipy
import subprocess

from midiutil.MidiFile import MIDIFile

# MIDI and WAV utils
def write_wav(midi_data, filename, tempo=240):
    write_midi_file(midi_data, "_tmp.mid", tempo=tempo)
    midi_to_wav("_tmp.mid", filename)
    subprocess.run(["rm", "_tmp.mid"], stdout = subprocess.DEVNULL, stderr = subprocess.DEVNULL)


def write_mp3(midi_data, filename, tempo=240):
    write_midi_file(midi_data, "_tmp.mid", tempo=tempo)
    midi_to_mp3("_tmp.mid", filename)
    subprocess.run(["rm", "_tmp.mid"], stdout = subprocess.DEVNULL, stderr = subprocess.DEVNULL)


def midi_to_wav(midi_path, wav_path):
    subprocess.run(["timidity", midi_path, "-Ow", "-o", wav_path], stdout = subprocess.DEVNULL, stderr = subprocess.DEVNULL)


def midi_to_mp3(midi_path, mp3_path):
    subprocess.run([f"timidity {midi_path} -Ow -o - | ffmpeg -i - -acodec libmp3lame -ab 64k {mp3_path}"], shell=True, stdout = subprocess.DEVNULL, stderr = subprocess.DEVNULL)

def write_midi_file(midi_data, filename, track=0, time_offset=0, channel=0, tempo=240, volume=100, duration=1):
    """
        midi_data: [[(midi_key, quarter_length), ...], ...]
    
    """
    dir = os.path.dirname(filename)
    if dir: os.makedirs(dir, exist_ok=True)

    midi = MIDIFile(1)
    midi.addTempo(track, time_offset, tempo)
    
    for part in midi_data:
        time_offset = 0
        for (midi_key, quarter_length) in part:
            if midi_key != 0:
                midi.addNote(track, channel, midi_key, time_offset, duration, volume)
            
            time_offset += quarter_length

    with open(filename, "wb") as f:
        midi.writeFile(f)


# MIDI conversion utils

NOTES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
OCTAVES = list(range(11))
NOTES_IN_OCTAVE = len(NOTES)

def note_to_number(note: str, octave: int) -> int:
    assert note in NOTES
    assert octave in OCTAVES

    note = NOTES.index(note)
    note += (NOTES_IN_OCTAVE * octave)

    assert 0 <= note <= 127

    return note

def number_to_note(number: int) -> tuple:
    octave = number // NOTES_IN_OCTAVE
    assert octave in OCTAVES
    assert 0 <= number <= 127
    note = NOTES[number % NOTES_IN_OCTAVE]

    return note, octave


# Plotting utils
def plot_sequence(x_sequence, state_sequence, dots=False):
    fig, ax = plt.subplots(figsize=(12,4))

    color_palette = distinctipy.get_colors(np.max(state_sequence) + 1)
    colors = [color_palette[s] for s in state_sequence]

    xs = np.arange(x_sequence.shape[0])
    cs = np.linspace(np.min(x_sequence), np.max(x_sequence), 12)
    ax.hlines(cs, 0, xs.shape[0], linestyles="dotted", color="gray")
    
    if dots: 
        ax.scatter(xs, x_sequence, color=colors)
    else:
        for i in range(x_sequence.shape[0]):
            if x_sequence[i] == 0: continue
            x, y = xs[i], x_sequence[i]
            note = number_to_note(y)[0]
            ax.text(x-.1, y-.22, note, color=colors[i], size=12)

    ax.set_ylabel("note")
    ax.set_xlabel("time")
    ax.set_title("Melody sequence")
    ax.set_xlim(0, x_sequence.shape[0])


# Misc utils
def fix_time(part, quarter_length=1):
    # [note1, ..., note_n] -> [(note1, quarter_length), ..., (note_n, quarter_length)]
    return [(note, quarter_length) for note in part]


def audio_widget(path, ignore=True):
    # ignore=True to reduce the size of the notebook on git
    if ignore: return
    import IPython.display as ipd
    return ipd.Audio(path)