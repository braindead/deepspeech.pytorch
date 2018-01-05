import os.path
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'deepspeech.pytorch'))

import re
from math import exp
import argparse
import json
import subprocess
import tempfile
import time

import torch
from torch.autograd import Variable

import warnings
warnings.simplefilter("ignore", UserWarning)

from data.data_loader import SpectrogramParser
from decoder import GreedyDecoder, BeamCTCDecoder
from model import DeepSpeech as DeepSpeechModel

import diff_match_patch
from merge_ctms import merge_ctms
from common_functions import sox_trim, get_duration, convert_to_wav

MAX_SECONDS = 20
OVERLAP_SECONDS = 2
SECONDS_PER_TIMESTEP = 0.02004197271773347324
MAX_SEQ_LENGTH = 4001

class DeepSpeech():
    def __init__(self, model_path, lm_path='', trie_path='', lm_alpha=0.5, lm_beta1=0.8, lm_beta2=0.8, beam_width=128, cuda=False, decoder='beam'):
        self.decoder_type = decoder

        self.model = DeepSpeechModel.load_model(model_path, cuda=cuda)
        self.model.eval()

        self.labels = DeepSpeechModel.get_labels(self.model).lower()
        self.audio_conf = DeepSpeechModel.get_audio_conf(self.model)
        
        self.parser = SpectrogramParser(self.audio_conf, normalize=True)

        self.int_to_char = dict([(i, c) for (i, c) in enumerate(self.labels)])

        if self.decoder_type == "beam":
            self.decoder = BeamCTCDecoder(self.labels, lm_path=lm_path, alpha=lm_alpha, beta=lm_beta1, cutoff_top_n=40, cutoff_prob=1.0, beam_width=beam_width, num_processes=4)
        else:
            self.decoder = GreedyDecoder(self.labels, blank_index=self.labels.index('_'))

    def chars_to_word(self, chars):
        word = re.sub(r"([a-z_'])\1{1,}", r"\1", ''.join(chars))
        word = re.sub('_', '', word)
        return word.strip()

    def finalize_ctm(self, chars, probabilities, start_ts):
        word = self.chars_to_word(chars)
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

        conf = round(sum(probabilities)/len(probabilities), 2)
        start = round(start_ts * SECONDS_PER_TIMESTEP, 3)
        end = round(end_ts * SECONDS_PER_TIMESTEP, 3)
        duration = round((end_ts - start_ts) * SECONDS_PER_TIMESTEP, 2)

        return {'chars': ''.join(chars), 'word': word, 'conf': conf, 'start': start, 'end': end, 'duration': duration}

    def build_ctms(self, probs, sizes):
        _, max_probs = torch.max(probs.transpose(0, 1), 2)

        ctms = []
        for segment in range(probs.size(1)):
            segment_ctms = []
            chars = []
            start_ts = None
            probabilities = []
            last_char = ''

            for i in range(sizes[segment]):
                char = self.int_to_char[max_probs[segment][i]];

                if start_ts == None:
                    start_ts = i

                chars.append(char)
                probabilities.append(max(probs[i][segment]))

                if char == ' ' and last_char != ' ':
                    ctm = self.finalize_ctm(chars, probabilities, start_ts)
                    if ctm != None:
                        segment_ctms.append(ctm)
                    chars = []
                    probabilities = []
                    start_ts = None

                last_char = char

            if len(chars) > 0:
                ctm = self.finalize_ctm(chars, probabilities, start_ts)
                if ctm != None:
                    segment_ctms.append(ctm)

            ctms.append(segment_ctms)

        return ctms

    def parse_audio(self, audio_path):
        return self.parser.parse_audio(audio_path)

    def segment(self, audio_path):
        duration = get_duration(audio_path)

        spects = []
        start_time = 0
        remaining = duration

        while start_time < duration:
            segment_len = MAX_SECONDS
            if start_time + segment_len > duration:
                segment_len = duration - start_time

            segment = tempfile.NamedTemporaryFile(suffix='.wav', delete=False).name
            sox_trim(audio_path, segment, start_time, segment_len)

            spect = self.parse_audio(segment)
            spects.append(spect)

            os.remove(segment)
            start_time += MAX_SECONDS - OVERLAP_SECONDS

        return spects

    def combine(self, ctms):
        combined = []

        for segment in range(len(ctms)):
            segment_ctms = ctms[segment]

            next_segment_ctms = []
            if segment + 1 < len(ctms):
                next_segment_ctms = ctms[segment + 1]

            segment_start = segment * (MAX_SECONDS - OVERLAP_SECONDS)
            segment_end = segment_start + MAX_SECONDS
            if len(next_segment_ctms) > 0:
                segment_end -= OVERLAP_SECONDS

            for ctm in segment_ctms:
                if 'skip' in ctm:
                    continue

                ctm['start'] += segment_start
                ctm['end'] += segment_start

                if ctm['start'] <= segment_end:
                    combined.append(ctm)
                else:
                    match_found = False
                    overlap_ctms = [c for c in next_segment_ctms if c['start'] <= OVERLAP_SECONDS]
                    for c in overlap_ctms:
                        if ctm['word'] == c['word']:
                            match_found = True
                            break
                        else:
                            c['skip'] = 1

                    if match_found == True:
                        break

        return combined

    def predict(self, spects, batch_size):
        ctms = []

        processed = 0
        num_segments = len(spects)

        while processed < num_segments:
            if processed + batch_size > num_segments:
                batch_size = num_segments - processed

            inputs = torch.zeros(batch_size, 1, spects[0].size(0), MAX_SEQ_LENGTH)
            input_percentages = torch.FloatTensor(batch_size)
            for i, spect in enumerate(spects[processed:processed+batch_size]):
                input_percentages[i] = spect.size(1)/MAX_SEQ_LENGTH
                inputs[i][0].narrow(1, 0, spect.size(1)).copy_(spect)

            out = self.model(Variable(inputs, volatile=True))
            out = out.transpose(0, 1)  # TxNxH
            seq_length = out.size(0)
            sizes = input_percentages.mul_(int(seq_length)).int()
            decoded_output, _ = self.decoder.decode(out.data, sizes)

            batch_ctms = self.build_ctms(out.data, sizes)
            if self.decoder_type == "beam":
                for i, ctm in enumerate(batch_ctms):
                    merged = merge_ctms(ctm, decoded_output[i][0])
                    batch_ctms[i] = merged

            ctms.extend(batch_ctms)
            processed += batch_size
            print("Predicted {0:.2f}% data".format(100*float(processed)/num_segments), end="\r")

        #ctms = self.combine(ctms)

        return ctms

    def predict_from_logits(self, logits):
        ctms = []
        num_batches = logits.shape[0]

        for i in range(num_batches):
            out = torch.from_numpy(logits[i][0])
            sizes = torch.from_numpy(logits[i][1])
            decoded_output, _ = self.decoder.decode(out, sizes)

            batch_ctms = self.build_ctms(out, sizes)
            if self.decoder_type == "beam":
                for i, ctm in enumerate(batch_ctms):
                    merged = merge_ctms(ctm, decoded_output[i][0])
                    batch_ctms[i] = merged
            ctms.extend(batch_ctms)

        return ctms

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DeepSpeech prediction')
    parser.add_argument('--model_path', default='models/phoenix32.pth.tar', help='Path to model file created by training')
    parser.add_argument('--audio_path', default='audio.wav', help='Audio file to predict on')
    parser.add_argument('--cuda', action="store_true", default=True, help='Use cuda for decoding')
    parser.add_argument('--debug', action="store_true", default=False,  help='print debug logs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size for GPU batching')
    parser.add_argument('--decoder', default="beam", choices=["greedy", "beam"], type=str, help="Decoder to use")
    parser.add_argument('--out_format', type=str, dest='out_format', default="json", choices=["json", "txt"], help='output format, txt or json are supported')

    beam_args = parser.add_argument_group("Beam Decode Options", "Configurations options for the CTC Beam Search decoder")
    beam_args.add_argument('--beam_width', default=128, type=int, help='Beam width to use')
    beam_args.add_argument('--lm_path', default="scribie-full-res.bin", type=str, help='Path to an (optional) kenlm language model for use with beam search (req\'d with trie)')
    beam_args.add_argument('--trie_path', default="scribie-full-res.trie", type=str, help='Path to an (optional) trie dictionary for use with beam search (req\'d with LM)')
    beam_args.add_argument('--lm_alpha', default=0.5, type=float, help='Language model weight')
    beam_args.add_argument('--lm_beta1', default=0.8, type=float, help='Language model word bonus (all words)')
    beam_args.add_argument('--lm_beta2', default=0.8, type=float, help='Language model word bonus (IV words)')
    args = parser.parse_args()

    if not os.path.isfile(args.model_path):
        raise Exception("{} not found".format(args.model_path))

    if not os.path.isfile(args.audio_path):
        raise Exception("{} not found".format(args.audio_path))

    if not os.path.isfile(args.lm_path):
        raise Exception("{} not found".format(args.lm_path))

    if not os.path.isfile(args.trie_path):
        raise Exception("{} not found".format(args.trie_path))

    t0 = time.time()

    wav_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False).name
    convert_to_wav(args.audio_path, wav_file)
    
    model = DeepSpeech(model_path=args.model_path, lm_path=args.lm_path, trie_path=args.trie_path, lm_alpha=args.lm_alpha, lm_beta1=args.lm_beta1, lm_beta2=args.lm_beta2, beam_width=args.beam_width, cuda=args.cuda, decoder=args.decoder)
    spects = model.segment(wav_file)
    ctms = model.predict(spects, args.batch_size)

    t1 = time.time()

    if args.out_format == "json":
        print(json.dumps(ctms))
    elif args.out_format == "txt":
        print(' '.join(c['word'] for c in ctms))

    if args.debug:
        duration = get_duration(wav_file)
        print("decoded {:.2f} seconds of audio in {:.2f} seconds, ratio {:.2f}".format(duration, t1-t0, (t1-t0)/duration))

    os.remove(wav_file)
