import os
import re
from pathlib import Path


class DebattenDataCleaner:
    """
    Takes raw html files and extracts the sentences and their timestamps.
    """

    # Initialises class and input, output locations
    def __init__(self, raw_subtitles_dir: Path, cleaned_subtitles_dir: Path):
        self.raw_subtitles_dir = raw_subtitles_dir
        self.cleaned_subtitles_dir = cleaned_subtitles_dir

    def setRawFilesLocation(self, new_loc):
        self.raw_subtitles_dir = new_loc

    def setProcessedFilesLocation(self, new_loc):
        self.cleaned_subtitles_dir = new_loc

    def getFileLocation(self, disp=True):

        if disp:
            if self.raw_subtitles_dir is None:
                print('Raw subtitles are not specified!')
            else:
                print('Raw subtitles are loaded from "{:s}"'.format(str(self.raw_subtitles_dir)))

            if self.cleaned_subtitles_dir is None:
                print('Save location for processed subtitles is not specified!')
            else:
                print('Save location for processed subtitles is "{:s}"'.format(str(self.cleaned_subtitles_dir)))

        return self.raw_subtitles_dir, self.cleaned_subtitles_dir

    # Returns the full (or relative) path all the raw files
    def getRawFilePaths(self, filename='page.html', subset=None):

        program_ids = os.listdir(str(self.raw_subtitles_dir))
        files = [str(Path(self.raw_subtitles_dir, p_id, filename)) for p_id in program_ids]

        if subset and abs(subset) < len(files):
            if subset < 0:
                files = files[subset:]
                program_ids = program_ids[subset:]
            else:
                files = files[:subset]
                program_ids = program_ids[:subset]

        return files, program_ids

    def html_decode(self, s):
        """
        Returns the ASCII decoded version of the given HTML string. This does
        NOT remove normal HTML tags like <p>.
        """
        htmlCodes = (
            ("'", '&#39;'),
            ('"', '&quot;'),  # ("'", '&quot;'),#
            ('>', '&gt;'),
            ('<', '&lt;'),
            ('&', '&amp;')
        )
        for code in htmlCodes:
            s = s.replace(code[1], code[0])
        return s

    # Extract the relevant HTML text for each sentence
    def getHtmlSentences(self, file_loc, re_subs_pattern='category="Undertekster"[\S\W<>]+<script src="/',
                         split_pattern='<span class="digits ng-binding">'):
        subtitle_file = open(file_loc, 'r', encoding='iso-8859-1')
        # subtitle_file = open(file_loc,'r', encoding='utf-8', errors='ignore')
        doc = subtitle_file.read()
        doc = self.html_decode(doc)

        subtitle_file.close()

        # Find part of the html which contain the subtitles
        p_subs = re.compile(re_subs_pattern)
        match = re.search(p_subs, doc)
        doc_subs = match.group()

        # Split into the sentences
        sentences = doc_subs.split(split_pattern)
        sentences = [sentences[i] for i in range(1, len(sentences))]

        print('\tProgram has {:d} sentences'.format(len(sentences)))
        return sentences

    def getTimeAndText(self, text, re_time='[\d:]+', re_text='ma-highlight="[\w?%!:-; \',.\d-]+'):
        p_time = re.compile(re_time)
        p_text = re.compile(re_text)

        text_at_this_timepoint = re.finditer(p_text, text)
        s = ''
        for match in text_at_this_timepoint:
            s = s + ' ' + (match.group()[14:])

        s = re.sub(' +', ' ', s)  # Removes repeating whitespaces
        s = s[1:]  # Removes first white space

        nums = re.finditer(p_time, text)
        nums = [n.group() for n in nums]

        time_start = nums[0]
        time_end = nums[1]

        return s, time_start, time_end

    def getCleanedSentences(self, sentences):
        program = []
        start_times = []
        end_times = []

        for sen in sentences:
            text, start, end = self.getTimeAndText(sen)

            if len(program) is 0:
                program.append(text)
                start_times.append(start)
                end_times.append(end)
            elif len(text) > 0:
                try:

                    if text[0] is '-':
                        last_text = program.pop()
                        s = last_text + text
                        s = re.sub('- ', ' ', s)
                        s = re.sub(' -', ' ', s)
                        s = re.sub(' +', ' ', s)

                        program.append(s)

                        end_times.pop()
                        end_times.append(end)

                    else:
                        program.append(text)
                        start_times.append(start)
                        end_times.append(end)

                except Exception as e:
                    print('Woops.. ')
                    print('"', text, '"')
                    print(len(text))

        print('\tProgram has {:d} cleaned sentences'.format(len(program)))
        return dict(zip(['sentences', 'start time', 'end time'], [program, start_times, end_times]))

    def getCleanedPrograms(self, file_paths, program_ids):
        total_sent = 0
        all_programs = dict()
        for i in range(len(file_paths)):
            print('Program {:d} of {:d} ({:s})'.format(i + 1, len(file_paths), program_ids[i]))
            program_sentences = self.getHtmlSentences(file_paths[i])
            all_programs[program_ids[i]] = self.getCleanedSentences(program_sentences)
            total_sent += len(all_programs[program_ids[i]]['sentences'])

        print('\nA total of {:d} sentences was found.'.format(total_sent))

        return all_programs

    def export_programs(self, all_programs):

        for p_id in all_programs:
            sentence_id = 1
            with Path(self.cleaned_subtitles_dir, 'program{}.html'.format(p_id)).open("w") as f:
                f.write('<span id="program ' + str(p_id) + '">\n')

                for s in all_programs[p_id]['sentences']:
                    f.write('\t<p id="' + str(p_id) + '"> ' + s + ' </p>\n')
                    sentence_id += 1

                f.write('</span>')
            print('Sucessfully wrote program' + str(p_id))
