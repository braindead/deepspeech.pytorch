import diff_match_patch

GUARD_SEC = 0.01
MAX_SECS_PER_CHAR = 0.5

def merge_ctms(ctms, text):
    ctm_text = ' '.join([c['word'] for c in ctms])

    if len(ctm_text.strip()) == 0:
        return []

    dmp = diff_match_patch.diff_match_patch()
    summary = dmp.diff_summary(ctm_text, text)

    ctm_index = 0
    for change in summary:
        change_type = change['type']
        text = change['text']
        orig_words = text[0].split()
        diff_ctms = []

        if change_type == "equal":
            ctm_index += len(orig_words)
            continue

        inserted_words = text[1].split()
        if len(inserted_words) > 0:
            diff_ctm_start_sec = 0
            diff_ctm_end_sec = 0

            if len(orig_words) == len(inserted_words):
                diff_ctm_start_sec = ctms[ctm_index]['start']
                diff_ctm_end_sec = ctms[ctm_index + len(orig_words) - 1]['end']
            else:
                if ctm_index - 1 >= 0:
                    ctm = ctms[ctm_index - 1]
                    diff_ctm_start_sec = ctm['start'] + ctm['duration'] + GUARD_SEC

                end_index = ctm_index + len(orig_words)
                if end_index >= len(ctms):
                    end_index = len(ctms) - 1

                diff_ctm_end_sec = ctms[end_index]['end'] - GUARD_SEC

            secs_per_char = (diff_ctm_end_sec - diff_ctm_start_sec)/(len(''.join(inserted_words)) + len(inserted_words)*GUARD_SEC)

            if secs_per_char < 0:
                secs_per_char = 0
            elif secs_per_char > MAX_SECS_PER_CHAR:
                available_gap = diff_ctm_end_sec - diff_ctm_start_sec

                secs_per_char = MAX_SECS_PER_CHAR
                adjusted_gap = len(''.join(inserted_words)) * secs_per_char

                diff_ctm_start_sec += available_gap - adjusted_gap - GUARD_SEC

            start = diff_ctm_start_sec
            for i, word in enumerate(inserted_words):
                duration = len(word) * secs_per_char
                ctm = {'start': round(start, 3), 'duration': round(duration, 2), 'end': round(start + duration, 3), 'word': word}

                if len(orig_words) == len(inserted_words):
                    ctm['conf'] = ctms[ctm_index + i]['conf']
                    ctm['chars'] = ctms[ctm_index + i]['chars']
                else:
                    ctm['conf'] = ctms[ctm_index]['conf']
                    ctm['chars'] = ctms[ctm_index]['chars']

                diff_ctms.append(ctm)
                start += duration + GUARD_SEC

        ctms = ctms[:ctm_index] + diff_ctms + ctms[(ctm_index + len(orig_words)):]
        ctm_index += len(diff_ctms)

    return ctms
