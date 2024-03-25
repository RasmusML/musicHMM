import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
import distinctipy

from midiutil.MidiFile import MIDIFile

# Data source:
# https://github.com/czhuang/JSB-Chorales-dataset/
def load_jsb_chorales(path="data/jsb-chorales-quarter.pkl"):
    with open(path, "rb") as f:
        raw = pickle.load(f)

    result = []
    for song_set in raw.values():
        for song in song_set:
            melody = [notes[0] if len(notes) > 0 else 0 for notes in song]
            result.append(melody)

    return result


def write_midi_file(midi_data, filename, track=0, time_offset=0, channel=0, tempo=240, volume=100, duration=1):
    dir = os.path.dirname(filename)
    if dir: os.makedirs(dir, exist_ok=True)

    midi = MIDIFile(1)
    midi.addTempo(track, time_offset, tempo)
    
    for i, note in enumerate(midi_data):
        if note == 0: continue
        midi.addNote(track, channel, note, time_offset + i, duration, volume)

    with open(filename, "wb") as f:
        midi.writeFile(f)


def write_midi_file2(midi_data, filename, track=0, time_offset=0, channel=0, tempo=240, volume=100, duration=1):
    dir = os.path.dirname(filename)
    if dir: os.makedirs(dir, exist_ok=True)

    midi = MIDIFile(1)
    midi.addTempo(track, time_offset, tempo)
    
    for i, (note, time) in enumerate(midi_data):
        if note == 0: continue
        midi.addNote(track, channel, note, time_offset + time, duration, volume)

    with open(filename, "wb") as f:
        midi.writeFile(f)


NOTES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
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
            x, y = xs[i], x_sequence[i]
            note = number_to_note(y)[0]
            ax.text(x-.1, y-.22, note, color=colors[i], size=12)

    ax.set_ylabel("note")
    ax.set_xlabel("time")
    ax.set_title("Melody sequence")
    ax.set_xlim(0, x_sequence.shape[0])
