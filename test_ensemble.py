import sys
import os.path
import argparse
import json
from DeepSpeech import DeepSpeech
import kenlm
import diff_match_patch
import tempfile
import time
from common_functions import get_duration, convert_to_wav
import re
import numpy as np

def choose(lang_model, words, variants):
    string = ' '.join(words) + ' '
    score1 = lang_model.score(string + variants[0], bos=False, eos=False)
    score2 = lang_model.score(string + variants[1], bos=False, eos=False)

    if score1 > score2:
        return 0
    else:
        return 1

def ensemble(lang_model, ctms):
    preds = []
    for _ctms in ctms:
        pred = ' '.join([c['word'] for c in _ctms])
        preds.append(pred)

    dmp = diff_match_patch.diff_match_patch()
    summary = dmp.diff_summary(preds[0], preds[1])

    result = []
    words = []
    offsets = [0, 0]

    for change in summary:
        change_type = change['type']
        text = change['text']

        if change_type == "equal":
            selected = 0
        else:
            selected = choose(lang_model, words[-5:], text)

        selected_words = text[selected].split()
        selected_len = len(selected_words)
        words.extend(selected_words)
        result.extend(ctms[selected][offsets[selected]:offsets[selected]+selected_len])

        if change_type == 'equal':
            offset_len = len(text[0].split())
            offsets[0] += offset_len
            offsets[1] += offset_len
        else:
            offsets[0] += len(text[0].split())
            offsets[1] += len(text[1].split())

    return result

def add_speaker_turns(ctms, turn_ctms):
    turns = [c['start'] for c in turn_ctms if '|' in c['chars']]

    last_ctm_index = 0
    for second in turns:
        for i, ctm in enumerate(ctms[last_ctm_index:], last_ctm_index):
            if ctm['start'] >= second and i > 0:
                ctms[i-1]['turn'] = 1
                last_ctm_index = i
                break

    return ctms

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DeepSpeech ensemble prediction')
    parser.add_argument('--stt_model_paths', type=lambda s: [m for m in s.split(',')], default='finch15.pth.tar,phoenix32.pth.tar', help='Paths speech to text models')
    parser.add_argument('--manifest_path', default='test.csv', help='Audio file to predict on')
    parser.add_argument('--cuda', action="store_true", default=False, help='Use cuda for decoding')
    parser.add_argument('--print_logs', action="store_true", default=False,  help='print debug logs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size for GPU batching')
    parser.add_argument('--force_predictions', action='store_true', default=False, help="Predictions would be run on each model if set true, else the .npy logits will be used")
    parser.add_argument('--logits', type=lambda s: [m for m in s.split(',')], default='logits/finch_latest.npy,logits/phoenix_latest.npy', help='Logits which are previously generated outputs from a model') 

    beam_args = parser.add_argument_group("Beam Decode Options", "Configurations options for the CTC Beam Search decoder")
    beam_args.add_argument('--beam_width', default=128, type=int, help='Beam width to use')
    beam_args.add_argument('--lm_path', default="default", type=str, help='Path to an kenlm language model for use with beam search')
    beam_args.add_argument('--lm_alphas', default="0.5,0.45", type=lambda s: [float(m) for m in s.split(',')], help='Language model weight')
    beam_args.add_argument('--lm_betas', default="1.35,1.45", type=lambda s: [float(m) for m in s.split(',')], help='Language model word bonus (all words)')
    args = parser.parse_args()

    if args.lm_path == 'default':
        raise Exception("please give path to the lamnguage model which should be used")

    if len(args.stt_model_paths) != 2:
        raise Exception("at least two speech to text models are required")

    if args.logits and not args.force_predictions:
        print("\nNOTE: Predictions won't be run on model and previously generated .npy files will be used. If you wish to force predictions from model, use --force_predictions parameter")

    print("\nLoading Models")
    models = []
    model_names = []
    for i in range(len(args.stt_model_paths)):
        path = os.path.expanduser(args.stt_model_paths[i])

        name = os.path.basename(path.split('.')[0])
        model_names.append(name)

        if not os.path.isfile(path):
            raise Exception("{} not found".format(path))

        model = DeepSpeech(model_path=path, lm_path=args.lm_path, trie_path='', lm_alpha=args.lm_alphas[i], lm_beta1=args.lm_betas[i], beam_width=32, cuda=args.cuda, decoder="beam")
        models.append(model)
        print("loaded model {0}".format(args.stt_model_paths[i]))

    lang_model = kenlm.Model(args.lm_path)

    if not args.logits or args.force_predictions:
        print("\nGenerating Spectograms and loading Transcripts")
    else:
        print("\nLoading transcripts")
    transcripts = []
    spects = []
    for line in open(args.manifest_path):
        audio_path, transcript_path = line.strip().split(',')
        tr = open(transcript_path).read().lower()
        tr = re.sub(" '", "'", tr)
        transcripts.append(tr)
        if not args.logits or args.force_predictions:
            spect = models[0].parse_audio(audio_path)
            spects.append(spect)

    if not args.logits or args.force_predictions:
        print("\nMaking Predictions")
    else:
        print("\nDecoding from logits")

    model_ctms = [[], []]
    for i, model in enumerate(models):
        if not args.logits or args.force_predictions:
            ctms = model.predict(spects, args.batch_size)
        else:
            logit = np.load(args.logits[i])
            ctms = model.predict_from_logits(logit)
        model_ctms[i] = ctms
        print("Predictions completed for model #{0}".format(i+1))

    print("\nCalculating WER and CER")
    wer, cer = 0, 0
    total_samples = len(model_ctms[0])
    for i in range(total_samples):
        ctms = ensemble(lang_model, [model_ctms[0][i], model_ctms[1][i]])
        transcript = ' '.join([c['word'] for c in ctms])
        '''
        print(transcript)
        transcript = ' '.join([c['word'] for c in model_ctms[0][i]])
        print(transcript)
        transcript = ' '.join([c['word'] for c in model_ctms[1][i]])
        print(transcript)
        '''
        reference = transcripts[i]

        wer += models[0].decoder.wer(transcript, reference) / float(len(reference.split()))
        cer += models[0].decoder.cer(transcript, reference) / float(len(reference))
        sys.stdout.write("Completed {0:.2f}%\r".format(100*float(i+1)/total_samples))

    print('\n\nTest Summary \t'
          'Average WER {wer:.3f}\t'
          'Average CER {cer:.3f}\t'.format(wer=wer/total_samples * 100, cer=cer/total_samples * 100))

