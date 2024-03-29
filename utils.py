import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
import distinctipy
import subprocess
import torch
import seaborn as sns

from midiutil.MidiFile import MIDIFile


def load_jsb_chorales(path="data/jsb-chorales-quarter.pkl"):
    with open(path, "rb") as f:
        raw = pickle.load(f)

    offset = -10

    result = []
    for song_set in raw.values():
        for song in song_set:
            melody = [list(np.array(notes) + offset) if len(notes) >= 1 else [] for notes in song]
            result.append(melody)

    return result


# MIDI and WAV utils
def write_wav(midi_data, filename, tempo=240):
    write_midi_file(midi_data, "_tmp.mid", tempo=tempo)
    midi_to_wav("_tmp.mid", filename)
    subprocess.run(["rm", "_tmp.mid"], stdout = subprocess.DEVNULL, stderr = subprocess.DEVNULL)


def write_mp3(midi_data, filename, tempo=240, volume=100):
    write_midi_file(midi_data, "_tmp.mid", tempo=tempo, volume=volume)
    midi_to_mp3("_tmp.mid", filename)
    subprocess.run(["rm", "_tmp.mid"], stdout = subprocess.DEVNULL, stderr = subprocess.DEVNULL)


def midi_to_wav(midi_path, wav_path):
    subprocess.run(["timidity", midi_path, "-Ow", "-o", wav_path], stdout = subprocess.DEVNULL, stderr = subprocess.DEVNULL)


def midi_to_mp3(midi_path, mp3_path):
    subprocess.run([f"timidity {midi_path} -Ow -o - | ffmpeg -i - -acodec libmp3lame -ab 64k {mp3_path}"], shell=True, stdout = subprocess.DEVNULL, stderr = subprocess.DEVNULL)

def write_midi_file2(midi_data, filename, track=0, time_offset=0, channel=0, tempo=240, volume=100, duration=1):
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


def write_midi_file(midi_data, filename, track=0, time_offset=0, channel=0, tempo=240, volume=100, duration=1):
    dir = os.path.dirname(filename)
    if dir: os.makedirs(dir, exist_ok=True)

    midi = MIDIFile(1)
    midi.addTempo(track, time_offset, tempo)
    
    time_offset = 0
    for chord in midi_data:
        for note in chord:
            if note == 0: continue
            midi.addNote(track, channel, note, time_offset, duration, volume)

        time_offset += 1

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
    

def str_number_to_note(number: int) -> str:
    note, octave = number_to_note(number)
    return f"{note}{octave}"


# Plotting utils
def plot_sequence(sequence_onehot, state_sequence):
    fig, ax = plt.subplots(figsize=(14,4))

    color_palette = distinctipy.get_colors(np.max(state_sequence) + 1)
    colors = [color_palette[s] for s in state_sequence]

    xs = np.arange(sequence_onehot.shape[0])

    all_idxs = np.argwhere(sequence_onehot).flatten()
    cs = np.linspace(np.min(all_idxs), np.max(all_idxs), 12)
    ax.hlines(cs, 0, xs.shape[0], linestyles="dotted", color="gray")

    # dots + vertical lines
    for i in range(sequence_onehot.shape[0]):
        idxs = np.argwhere(sequence_onehot[i]).flatten()
        if len(idxs) == 0: continue
        if len(idxs) == 1 and idxs[0] == 0: continue

        for j in idxs:
            x, y = xs[i], j
            ax.scatter(x, y, color=colors[i], s=80)

        ax.vlines(x, np.min(idxs), np.max(idxs), color=colors[i], linestyles="dotted")

    # note names
    for i in range(sequence_onehot.shape[0]):
        idxs = np.argwhere(sequence_onehot[i]).flatten()
        if len(idxs) == 0: continue
        if len(idxs) == 1 and idxs[0] == 0: continue

        for j in idxs:
            x, y = xs[i], j
            note = number_to_note(j)[0]
            ax.text(x-.2, y-1.5, note, color="black", size=11)

    ax.set_ylabel("note")
    ax.set_xlabel("time")
    ax.set_title("Sequence")
    ax.set_xlim(0, sequence_onehot.shape[0])


def plot_loss(train_losses, validation_losses):
    fig, ax = plt.subplots(1, 1, figsize=(4, 3))
    ax.plot(train_losses, label="train")
    ax.plot(validation_losses, label="validation")
    ax.set_title("Loss")
    ax.set_xlabel("Iterations")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid()


def heatmap_transition_matrix(transition_matrix, ax=None):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))

    hidden_dim = transition_matrix.shape[0]
    text = np.array([[f"{transition_matrix[i, j]:.1f}" for j in range(hidden_dim)] for i in range(hidden_dim)])
    sns.heatmap(transition_matrix, cmap="viridis", ax=ax, annot=text, fmt="s")
    ax.set_xlabel("To")
    ax.set_ylabel("From")
    ax.set_title("Transition matrix")

    return ax


def plot_hidden_states_histogram(hidden_states, hidden_dim, ax=None):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(4, 3))

    bins = np.arange(hidden_dim + 1)
    ax.hist(hidden_states, bins=bins, color="skyblue", edgecolor="black", align="left")
    ax.set_xlabel("Hidden state")
    ax.set_ylabel("Count")
    ax.set_title(f"Hidden states of sampled data (n={hidden_states.shape[0]})")
    ax.set_xticks(bins[:-1])
    ax.grid()
    ax.set_axisbelow(True)

    return ax


# Misc utils
def fix_time(part, quarter_length=1):
    # [note1, ..., note_n] -> [(note1, quarter_length), ..., (note_n, quarter_length)]
    return [(note, quarter_length) for note in part]


def audio_widget(path, ignore=False):
    # ignore=True to reduce the size of the notebook on git
    if ignore: return
    import IPython.display as ipd
    return ipd.Audio(path)


# One-hot encoding <-> midi note conversion
def idx_to_onehot(sequence_idx, data_dim=97):
    length = len(sequence_idx)

    sequences_hot = np.zeros((length, data_dim))

    for t, notes_pressed in enumerate(sequence_idx):
        sequences_hot[t, notes_pressed] = 1

    return sequences_hot


def one_hot_to_idx(sequence):
    return list(np.argwhere(sequence).flatten())


def idx_to_onehot_multi(sequences_idx, data_dim=97):
    n_sequences = len(sequences_idx)
    lengths = np.array([len(song) for song in sequences_idx])
    max_length = lengths.max()

    sequences_hot = np.zeros((n_sequences, max_length, data_dim))

    for i, sequence in enumerate(sequences_idx):
        for t, notes_pressed in enumerate(sequence):
            sequences_hot[i, t, notes_pressed] = 1

    return sequences_hot