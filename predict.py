import argparse
import sys
import time
from math import exp
import re

import torch
from torch.autograd import Variable

from data.data_loader import SpectrogramParser
from decoder import GreedyDecoder, BeamCTCDecoder, Scorer, KenLMScorer
from model import DeepSpeech
import diff_match_patch

parser = argparse.ArgumentParser(description='DeepSpeech prediction')
parser.add_argument('--model_path', default='models/deepspeech_final.pth.tar',
                    help='Path to model file created by training')
parser.add_argument('--audio_path', default='audio.wav',
                    help='Audio file to predict on')
parser.add_argument('--cuda', action="store_true", help='Use cuda to test model')
parser.add_argument('--decoder', default="greedy", choices=["greedy", "beam"], type=str, help="Decoder to use")
beam_args = parser.add_argument_group("Beam Decode Options", "Configurations options for the CTC Beam Search decoder")
beam_args.add_argument('--beam_width', default=10, type=int, help='Beam width to use')
beam_args.add_argument('--lm_path', default=None, type=str, help='Path to an (optional) kenlm language model for use with beam search (req\'d with trie)')
beam_args.add_argument('--trie_path', default=None, type=str, help='Path to an (optional) trie dictionary for use with beam search (req\'d with LM)')
beam_args.add_argument('--lm_alpha', default=0.8, type=float, help='Language model weight')
beam_args.add_argument('--lm_beta1', default=1, type=float, help='Language model word bonus (all words)')
beam_args.add_argument('--lm_beta2', default=1, type=float, help='Language model word bonus (IV words)')
args = parser.parse_args()

SECONDS_PER_TIMESTEP = 0.02004197271773347324

def chars_to_word(chars):
    word = re.sub(r"([a-z_'])\1{1,}", r"\1", ''.join(chars))
    word = re.sub('_', '', word)
    return word.strip()

def finalize_ctm(chars, probabilities, start_ts):
    word = chars_to_word(chars)
    if len(word) == 0:
        return None

    chars = ''.join(chars)

    leading_blanks = re.search("^[_ ]+", chars)
    if leading_blanks:
        count = len(leading_blanks.group(0))
        if count > 10:
            start_ts += count - 10

    end_ts = start_ts + len(chars)
    trailing_blanks = re.search("[_ ]+$", chars)
    if trailing_blanks:
        count = len(trailing_blanks.group(0))
        if count > 10:
            end_ts -= count - 10

    conf = float("{:.2f}".format(sum(probabilities)/len(probabilities)))
    start = float("{:.3f}".format(start_ts * SECONDS_PER_TIMESTEP))
    end = float("{:.3f}".format(end_ts * SECONDS_PER_TIMESTEP))
    duration = float("{:.2f}".format((end_ts - start_ts) * SECONDS_PER_TIMESTEP))

    return {'chars': ''.join(chars), 'word': word, 'conf': conf, 'start': start, 'end': end, 'duration': duration}

if __name__ == '__main__':
    model = DeepSpeech.load_model(args.model_path, cuda=args.cuda)
    model.eval()

    print(args.audio_path)

    labels = DeepSpeech.get_labels(model).lower()
    audio_conf = DeepSpeech.get_audio_conf(model)

    if args.decoder == "beam":
        scorer = None
        if args.lm_path is not None:
            scorer = KenLMScorer(labels, args.lm_path, args.trie_path)
            scorer.set_lm_weight(args.lm_alpha)
            scorer.set_word_weight(args.lm_beta1)
            scorer.set_valid_word_weight(args.lm_beta2)
        else:
            scorer = Scorer()
        decoder = BeamCTCDecoder(labels, scorer, beam_width=args.beam_width, top_paths=1, space_index=labels.index(' '), blank_index=labels.index('_'))
    else:
        decoder = GreedyDecoder(labels, space_index=labels.index(' '), blank_index=labels.index('_'))

    parser = SpectrogramParser(audio_conf, normalize=True)

    t0 = time.time()
    spect = parser.parse_audio(args.audio_path).contiguous()
    spect = spect.view(1, 1, spect.size(0), spect.size(1))
    out = model(Variable(spect, volatile=True))
    out = out.transpose(0, 1)  # TxNxH
    decoded_output = decoder.decode(out.data)
    t1 = time.time()

    probs = out.data
    _, max_probs = torch.max(probs.transpose(0, 1), 2)
    int_to_char = dict([(i, c) for (i, c) in enumerate(labels)])

    ctms = []
    chars = []
    start_ts = None
    probabilities = []
    last_char = ''
    for i in range(probs.size(0)):
        char = int_to_char[max_probs[0][i]];

        if start_ts == None:
            start_ts = i

        chars.append(char)
        probabilities.append(exp(max(probs[i][0])))

        if char == ' ' and last_char != ' ':
            ctm = finalize_ctm(chars, probabilities, start_ts)
            if ctm != None:
                ctms.append(ctm)
            chars = []
            probabilities = []
            start_ts = None

        last_char = char

    if len(chars) > 0:
        ctm = finalize_ctm(chars, probabilities, start_ts)
        if ctm != None:
            ctms.append(ctm)

    print(ctms)
    ctm_output = ' '.join([c['word'] for c in ctms])
    print(ctm_output)

    print(decoded_output[0])

    dmp = diff_match_patch.diff_match_patch()
    dmp.Diff_Timeout = 0
    diffs = dmp.diff_wordMode(ctm_output, decoded_output[0])
    
    print(diffs)

    print("Decoded {0:.2f} seconds of audio in {1:.2f} seconds".format(spect.size(3)*audio_conf['window_stride'], t1-t0), file=sys.stderr)
