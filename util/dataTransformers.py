from .baseClasses import BaseDataTransformer
from .dataClasses import DataPath, DICTPrograms, HTMLInput
import os
import re
import numpy as np
import pandas as pd
from collections import OrderedDict

# ==============================================================================
# Specific Concrete Data Transformers Implementations
# ==============================================================================

# Following list describes the available formats
#
#   FORMATName      : Description
#   ============================================================================
#   HTMLSource      : Path to downloaded HTML pages from data source
#   DICTPrograms    : A dict with program_id : sentences as elements
#   HTMLInput       : sentences formatted in HTML for the annotator program
#   TABLEData       : Pandas table of the complete data
#
# Tensors:
#   char            : char based representation
#   PoS             : Part-of-Speech tags
#   Xbow            : Bag-Of-Words
#   word2vec        : Word2vec



PROGRAM_ID_BUGGED_PROGRAMS = ['7308025','2294023','2315222','2337314','2359717',\
                       '2304494','2348260', '3411204', '3570949', '3662558']

class fromHTMLSourceToDICTPrograms(BaseDataTransformer):
    """
    Transforms raw html files into DICT of the sentences and timestamps.
    """

    # Initialises class and input, output locations
    def __init__(self):
        inputFormat="HTMLSource"
        outputFormat="DICTPrograms"
        super().__init__(inputFormat,outputFormat)

    def _transform(self, HTMLSource):
        files, program_id = self.getRawFilePaths(str(HTMLSource.data))
        cleanedProgramsDict = self.getCleanedPrograms(files, program_id)
        cleanedProgramsDict = self.removeOutlier(cleanedProgramsDict)

        return DICTPrograms(data=cleanedProgramsDict)

    # Auxillary Functions

    def removeOutlier(self, cleanedProgramsDict):
        all_programs=cleanedProgramsDict

        a=[(len(all_programs[program]['sentences']), program) for program in all_programs]
        a = [a[i][0] for i in range(len(a))]
        np.sort(a)

        check_program = []

        for program in all_programs:
            if len(all_programs[program]['sentences']) < 100 or\
               len(all_programs[program]['sentences']) > 1000:
                check_program.append(program)

        self.logger.info('There is something fishy about the following programs')
        self.logger.info(check_program)

        for program in check_program:
            paragraphs = all_programs[program]['sentences']
            self.logger.info('--------------------------- %s --------------------' % program)
            self.logger.info('Has %i paragraphs' % len(paragraphs))
            #self.logger.info(paragraphs)
            self.logger.info('--------------------------------------------------------')


        # '8571627' # Test program, Always invalid!
        # '8793533' # Brexit program, rolling subtitles, can be fixed!
        # '8905036' #- rolling subs
        # '8573626' # empty program, cant be fixed
        # '8975996' #- rolling
        # '9024801' #- rolling

        # remove_programs = ['8571627','8793533', '8905036', '8573626', '8975996', '9024801']

        for program in check_program: #remove_programs:
            del all_programs[program]

        return all_programs

    # Returns the full (or relative) path all the raw files
    def getRawFilePaths(self, loc_raw_subtitles, filename='page.html', subset=None):

        program_ids = os.listdir(loc_raw_subtitles)
        files = ['{:s}/{:s}/{:s}'.format(loc_raw_subtitles, p_id, filename) for p_id in program_ids]

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
                ('"', '&quot;'),#("'", '&quot;'),#
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
        subtitle_file = open(file_loc,'r', encoding='iso-8859-1')
        #subtitle_file = open(file_loc,'r', encoding='utf-8', errors='ignore')
        doc = subtitle_file.read()
        doc = self.html_decode(doc)

        subtitle_file.close()

        # Find part of the html which contain the subtitles
        p_subs = re.compile(re_subs_pattern)
        match = re.search(p_subs,doc)
        doc_subs = match.group()

        #Split into the sentences
        sentences = doc_subs.split(split_pattern)
        sentences = [sentences[i] for i in range(1,len(sentences))]

        self.logger.info('\tProgram has {:d} sentences'.format(len(sentences)))
        return sentences


    def getTimeAndText(self, text, re_time='[\d:]+', re_text='ma-highlight="[\w?%!:-; \',.\d-]+'):
        p_time = re.compile(re_time)
        p_text = re.compile(re_text)

        text_at_this_timepoint = re.finditer(p_text,text)
        s = ''
        for match in text_at_this_timepoint:
            s=s+' '+(match.group()[14:])

        s = re.sub(' +',' ',s) # Removes repeating whitespaces
        s=s[1:] # Removes first white space

        nums = re.finditer(p_time,text)
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
            elif len(text)>0:
                try:

                    if text[0] is '-':
                        last_text = program.pop()
                        s = last_text+text
                        s = re.sub('- ',' ',s)
                        s = re.sub(' -',' ',s)
                        s = re.sub(' +',' ',s)

                        program.append(s)

                        end_times.pop()
                        end_times.append(end)

                    else:
                        program.append(text)
                        start_times.append(start)
                        end_times.append(end)

                except Exception as e:
                    self.logger.warning('Woops.. ')
                    self.logger.warning('"',text,'"')
                    self.logger.warning(len(text))

        self.logger.info('\tProgram has {:d} cleaned sentences'.format(len(program)))
        return dict(zip(['sentences','start time','end time'],[program, start_times, end_times]))


    def getCleanedPrograms(self, file_paths, program_ids):
        total_sent = 0
        all_programs = dict()
        count = 1;
        for i in range(len(file_paths)):
            self.logger.info('Program {:d} of {:d} ({:s})'.format(i+1,len(file_paths),program_ids[i]))
            program_sentences = self.getHtmlSentences(file_paths[i])
            all_programs[program_ids[i]] = self.getCleanedSentences(program_sentences)
            total_sent += len(all_programs[program_ids[i]]['sentences'])

        self.logger.info('\nA total of {:d} sentences was found.'.format(total_sent))

        return all_programs



class fromDICTProgramsToHTMLInput(BaseDataTransformer):
    """
    Transforms DICT of the sentences and timestamps into HTML formatted
    sentences used as input files for highlighting
    """


    # Initializes class and input, output locations
    def __init__(self):
        inputFormat="DICTPrograms"
        outputFormat="HTMLInput"
        super().__init__(inputFormat,outputFormat)

    def _transform(self,DICTPrograms):
        HTMLInput=self.export_debatten_programs(DICTPrograms)
        return HTMLInput

    def export_debatten_programs(self, DICTPrograms):
        FormattedSentences={}
        programs=DICTPrograms.data
        for p_id in programs:
            sentenc_id = 1
            #with open(self.loc_pro_subtitles+'program'+str(p_id), 'w') as f:
            programString='<span id="program '+str(p_id)+'">\n'

            for s in programs[p_id]['sentences']:
                programString+='\t<p id="'+str(p_id)+'"> '+s+' </p>\n'
                sentenc_id+=1

            programString+='</span>'
            FormattedSentences[str(p_id)] = programString
            self.logger.info('Sucessfully formatted program '+str(p_id))

        return HTMLInput(data=FormattedSentences,opts=DICTPrograms.opts)



class fromHTMLAnnotatedToDICTAnnotated(BaseDataTransformer):
    """
    Transforms annotated programs into a numpy(?) matrix
    """

    # Initializes class and input, output locations
    def __init__(self):
        inputFormat="HTMLAnnotated"
        outputFormat="DataMatrix"
        super().__init__(inputFormat,outputFormat)

    def _transform(self,HTMLAnnotated):
        DICTAnnotated=DICTPrograms()
        data = self.getAllCleanedProgramSentences(HTMLAnnotated)
        DICTAnnotated.data=data
        return DICTAnnotated


    def getFilePaths(self,path):
        files = os.listdir(path)
        return [path+f for f in files]


    def getProgramAndSentences(self,f_path):
        """Gets the program id, sentences id and sentences from a document"""
        with open(f_path,'r') as f:
            doc = f.read()

        #Find program id
        m_program_id = re.compile('program[\d]+')
        m = re.search(m_program_id, doc)
        program_id = m.group()


        sentences = doc.split('<p ')
        m_sentence_id = re.compile('id="[\d]+">')

        # Finds the sentence ids and removes html stuff from the begining of each sentence
        sentences_id = []
        for i in range(len(sentences)):
            match = re.search(m_sentence_id, sentences[i])
            if not match:
                sentences[i] = None
            else:
                sentences_id.append(int(match.group()[4:-2]))

                start_from = sentences[i].find('>')+1
                sentences[i] = sentences[i][start_from:]

        sentences = list(filter(None, sentences)) # Remove None elements
        assert(len(sentences)==len(sentences_id))

        return program_id, sentences_id, sentences

    # Finds highligted text including its surrounding patttern
    def findHighlights(self,s):
        m_highlight = re.compile('<span id="highlight["\w\d ]+class="highlight[\w"]+>[\w\d. ,!?%]+</span>')
        return re.findall(m_highlight, s)

    # Extracts highlighted text only
    def extractHighlights(self, s_matches):#Extracted the text highlighted
        m_high_text = re.compile('">[\w\d ,.!?%]+</')
        high_text = [re.findall(m_high_text, s_matches[i])[0][2:-2] for i in range(len(s_matches))]
        return [s.lstrip().rstrip() for s in high_text]

    # Removes html tags (and crap) from the string.
    def cleanSentence(self, s, disp=False):

        m_crap = re.compile('<[\w\d "=/]+>')
        s_crap_free = s
        for pattern in re.findall(m_crap, s):
            self.logger.debug(pattern)
            s_crap_free = s_crap_free.replace(pattern,'')

        #s_crap_free = re.sub('id="[\d]+">','',s_crap_free) # only during dev

        s_crap_free = s_crap_free.replace('\t',' ') # removes tabs
        s_crap_free = re.sub(' +',' ', s_crap_free) # removes excess spaces
        return s_crap_free.lstrip().rstrip()

    def getHighlight_indices(self,s,s_highlighted):

        # Two heuristic for correcting partially highlighted words.
        def getLeadingSpace(s,start_idx):
            # Finds first leading space before index "start_idx" in s
            if start_idx < 0:
                return 0
            elif s[start_idx] is ' ' :
                return start_idx+1
            else:
                return getLeadingSpace(s,start_idx-1)

        def getTailingSpace(s,end_idx):
            # Finds first trailing space after index "end_idx" in s
            if end_idx >= len(s):
                return len(s)
            elif s[end_idx] is ' ' or end_idx == len(s):
                return end_idx
            else:
                return getTailingSpace(s,end_idx+1)

        # Find the indicies of highlighted words
        indices = []
        # Get matched indices
        for m in s_highlighted:
            m_pattern = re.compile(m)
            match = re.search(m_pattern, s)
            if match:
                indices.append([getLeadingSpace(s, match.start()),
                                getTailingSpace(s, match.end())])
            else:
                self.logger.debug(match)
                self.logger.debug(m)
                self.logger.debug(s_highlighted)
                self.logger.debug(s+'\n')

        #self.logger.info('\n\n')
        return indices

    def getCleanedProgramSentences(self, sentences):
        sentences_processed = [None]*len(sentences)
        sentences_highlight = [None]*len(sentences)
        sentences_highlight_ind = [None]*len(sentences)

        for i in range(len(sentences)):
            sen = sentences[i]
            raw_highlights = self.findHighlights(sen)
            text_highlights = self.extractHighlights(raw_highlights)

            #Crap free verion
            sentences_processed[i] = self.cleanSentence(sen)
            #self.logger.info('cleaned: '+sentences_processed[i])
            indices_highlights = self.getHighlight_indices(sentences_processed[i],
                                                                         text_highlights)
            sentences_highlight_ind[i] = indices_highlights

            for idx in indices_highlights:
                if sentences_highlight[i]:
                     sentences_highlight[i] = sentences_highlight[i]+ ' [new claim]: '\
                                              +sentences_processed[i][idx[0]:idx[1]]
                else:
                    sentences_highlight[i] = sentences_processed[i][idx[0]:idx[1]]


        return sentences_processed, sentences_highlight, sentences_highlight_ind

    # EXPERIMENTAL!!! Processing multi-claim paragraphs
    def processMultiClaim(self,s,idx):
        merge_claims = []
        for c in range(len(idx)-1):
            if abs(idx[c][1]-idx[c+1][0]) == 1: #It is the same claim
                merge_claims.append(True)
            else:
                merge_claims.append(False)

        new_s = []
        new_idx = []
        for c in range(len(idx)-1):
            if merge_claims[c]:
                start_id = idx[c][0]
                end_id = idx[c+1][1]
                new_idx.append([start_id, end_id])
                new_s.append(s[start_id:end_id])
            else:
                if c > 0:
                    new_s.append(' [new claim]: ')

                start_id = idx[c][0]
                end_id = idx[c][1]
                new_idx.append([start_id, end_id])
                new_s.append(s[start_id:end_id])

        if not merge_claims[-1]:

            new_s.append(' [new claim]: ')

            start_id = idx[-1][0]
            end_id = idx[-1][1]
            new_idx.append([start_id, end_id])
            new_s.append(s[start_id:end_id])


        new_s = ''.join(new_s)
        return new_s, new_idx

    def getAllCleanedProgramSentences(self,HTMLAnnotated,disp=False):
        file_paths = self.getFilePaths(str(HTMLAnnotated.data))

        all_program_id = [None]*len(file_paths)
        all_sentences = [None]*len(file_paths)
        all_sentences_id = [None]*len(file_paths)
        all_highlights = [None]*len(file_paths)
        all_highlights_ind = [None]*len(file_paths)

        total_claims = 0;
        total_sentences = 0;

        for f in range(len(file_paths)):
            all_program_id[f], all_sentences_id[f], sentences = \
            self.getProgramAndSentences(file_paths[f])
            self.logger.info('Program id {:s}'.format(all_program_id[f]))

            all_sentences[f], all_highlights[f], all_highlights_ind[f] = \
                        self.getCleanedProgramSentences(sentences)

            num_claims = len(list(filter(None,all_highlights[f])))
            self.logger.info('\tThere were {:d} claims out of {:d} sentences ({:2.2f}%)'.format(num_claims\
                    ,len(sentences), num_claims/float(len(sentences))*100))

            total_claims = total_claims+num_claims
            total_sentences = total_sentences + len(sentences)

        self.logger.info('\nIn total there were {:d} claims out of {:d} sentences ({:2.2f}%)'.format(total_claims\
                , total_sentences, total_claims/float(total_sentences)*100))

        # ...
        labels = ['program_id', 'sentence_id', 'sentence', 'claim_idx', 'claim']

        data = {}

        bugged_programs = {'program1':'7308025',
                           'program2':'2294023',
                           'program3':'2315222',
                           'program4':'2337314',
                           'program5':'2359717',
                           'program6':'2304494',
                           'program7':'2348260',
                           'program8':'3411204',
                           'program9':'3570949',
                           'program10':'3662558'
                           }

        for pi, program_id in enumerate(all_program_id):
            if "program" in program_id:
                program_id = bugged_programs[program_id]
            self.logger.info(program_id)
            self.logger.info(pi)
            data[program_id] = {}
            data[program_id]['sentence_id'] = []
            data[program_id]['sentences'] = []
            data[program_id]['claim_idx'] = []
            data[program_id]['claim'] = []

            for si, sentence in enumerate(all_sentences[pi]):

                data[program_id]['sentence_id'].append(all_sentences_id[pi][si])
                data[program_id]['sentences'].append(all_sentences[pi][si])

                if len(all_highlights_ind[pi][si]) == 1:
                    data[program_id]['claim_idx'].append(all_highlights_ind[pi][si])
                    data[program_id]['claim'].append(all_highlights[pi][si])

                elif all_highlights_ind[pi][si]:

                    self.logger.info('HELP')
                    self.logger.info(all_program_id[pi])
                    #self.logger.info(all_sentences[p][si])
                    self.logger.info(all_highlights_ind[pi][si])
                    self.logger.info(all_highlights[pi][si])
                    new_s, new_idx = self.processMultiClaim(all_sentences[pi][si],\
                                                      all_highlights_ind[pi][si])

                    self.logger.info('Trying to handle this multi-claim, is the output correct?')
                    self.logger.info(new_idx)
                    self.logger.info(new_s)
                    # self.logger.info()

                    data[program_id]['claim_idx'].append(new_idx)
                    data[program_id]['claim'].append(new_s)
                else:
                    data[program_id]['claim_idx'].append(None)
                    data[program_id]['claim'].append(None)

        return data


class fromDICTProgramsToTABLEData(BaseDataTransformer):
    """
    Transforms annotated programs into a numpy(?) matrix
    """

    # Initializes class and input, output locations
    def __init__(self):
        inputFormat="DICTPrograms"
        outputFormat="TABLEData"
        super().__init__(inputFormat,outputFormat)

    def _transform(self,DICTPrograms):
        X, features=self.createTable(DICTPrograms)
        X=self.correctErrors(X,DICTPrograms.data['DICTPreannotated'])
        TABLEData=pd.DataFrame(X)
        TABLEData.columns=features
        return TABLEData


    def createTable(self,DICTPrograms):
        self.logger.debug(DICTPrograms.data)
        labels=list(DICTPrograms.data['DICTAnnotated'].data[
            next(iter(DICTPrograms.data['DICTAnnotated'].data.keys()))].keys())

        data=DICTPrograms.data['DICTAnnotated'].data
        dataPreannotated=DICTPrograms.data['DICTPreannotated'].data

        N = 0
        features = ['program_id', 'start time', 'end time']
        [labels.append(fea) for fea in features]
        features=labels


        start_times = []
        end_times = []
        processed_programs = []

        for program_id, program_data in data.items():
            if program_id not in PROGRAM_ID_BUGGED_PROGRAMS:
                for sentence in program_data['sentences']:
                    [start_times.append(start_time)
                     for start_time in dataPreannotated[program_id]['start time']]

                    [end_times.append(end_time)
                     for end_time in dataPreannotated[program_id]['end time']]
            else:
                    [start_times.append('NA')
                     for i in range(len(data[program_id]['sentences']))]

                    [end_times.append('NA')
                     for i in range(len(data[program_id]['sentences']))]

#            [start_times.append(dataPreannotated[program_id]['start time'][i])
#             if sentence.strip() in
#             map(str.strip, dataPreannotated[program_id]['sentences'])
#             else
#             self.logger.debug('{}: {}/{}, {} / {}\n'.format(program_id,len(program_data['sentences']),
#                                                 len(dataPreannotated[program_id]['sentences']),
#                                                 sentence, dataPreannotated[program_id]['sentences'][i]))
#             for i, sentence in enumerate(program_data['sentences'])]
#
#            [end_times.append(dataPreannotated[program_id]['end time'][i])
#             for i, sentence in enumerate(program_data['sentences'])
#             if sentence in dataPreannotated[program_id]['sentences']]
            N+=len(program_data['sentences'])
            processed_programs.append(program_id)
        frames=[]
        for key, program in data.items():
            d=self.flatten(program)
            d['program_id'] = [key]*len(d['claim'])
            frames.append(pd.DataFrame.from_dict(d))
        myTable=pd.concat(frames)
        # Sanity check
        # assert(len(start_times) == len(end_times))
        self.logger.debug(processed_programs)
        self.logger.debug(start_times[:10])
        self.logger.debug(len(start_times))
        self.logger.debug(N)
        self.logger.debug(type(myTable))
        self.logger.debug(myTable.shape)
        # assert(len(start_times) == N)

        #Concat data
        X = np.concatenate((myTable.as_matrix(),np.asarray([start_times,end_times]).T),axis=1)
        return X, features

    def flatten(self,d):
        '''
        Flatten an OrderedDict object
        '''
        result = OrderedDict()
        for k, v in d.items():
            if isinstance(v, dict):
                result.update(flatten(v))
            else:
                result[k] = v
        return result

    def correctErrors(self,X,DICTPrograms):
        # Convert the fake program names to real program ids


        ## FIX inconsistencies related to the inclusion of "-" in the paragraphs
        for program in PROGRAM_ID_BUGGED_PROGRAMS: # Fix each of the bugged programs

            idx_X = np.where(X[:,2] == program)[0] #Index in X

            for elem in range(idx_X.shape[0]): # For each paragraph

                X[idx_X[elem],2] = program

                para_bugged = X[idx_X[elem], 4]
                para_true = DICTPrograms[program]['sentences'][elem]
                # Replace the bugged sentence with the corrected one
                X[idx_X[elem], 4] = para_true


                if X[idx_X[elem],6]: # If there is a claim
                    #print(X[idx_X[elem],5])

                    if len(X[idx_X[elem],5]) == 1:
                        start_id = X[idx_X[elem],5][0][0]
                        end_id = X[idx_X[elem],5][0][1]

                        claim_idx = [self.getLeadingSpace(para_true,start_id),\
                                     self.getTailingSpace(para_true,end_id)]

                        X[idx_X[elem],5][0] = claim_idx
                    else:
                        claim_idx = []

                        print('Found:\n%s' %X[idx_X[elem],6])
                        print(X[idx_X[elem],5])

                        for c in range(len(X[idx_X[elem],5])):

                            start_id = X[idx_X[elem],5][c][0]
                            end_id = X[idx_X[elem],5][c][1]

                            claim_idx.append([self.getLeadingSpace(para_true,start_id),\
                                         self.getTailingSpace(para_true,end_id)])

                        for idx in claim_idx:
                            print(para_true[idx[0]:idx[1]]+'\n')

                        X[idx_X[elem],5] = claim_idx

                        print(para_true)

                        print(claim_idx)
        return X

    def getLeadingSpace(s,start_idx):
        # Finds first leading space before index "start_idx" in s
        if start_idx < 0:
            return 0
        elif s[start_idx] is ' ' :
            return start_idx+1
        else:
            return getLeadingSpace(s,start_idx-1)

    def getTailingSpace(s,end_idx):
        # Finds first trailing space after index "end_idx" in s
        if end_idx >= len(s):
            return len(s)
        elif s[end_idx] is ' ' or end_idx == len(s):
            return end_idx
        else:
            return getTailingSpace(s,end_idx+1)
