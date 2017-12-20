import os
import re
import numpy as np

class DebattenAnnotatedDatacleaner:
    """
    Takes the annotated programs, ...
    """
    
    # Initialises class and input, output locations
    def __init__(self, loc_ann=None, loc_out=None):
        self.loc_ann_subtitles = loc_ann if loc_ann is not None else []
        self.loc_out_subtitles = loc_out if loc_out is not None else []
    
    def setAnnotatedFilesLocation(self, new_loc):
        self.loc_ann_subtitles = new_loc
        
    def setOutputFilesLocation(self, new_loc):
        self.loc_out_subtitles = new_loc
    
    def getFileLocation(self, disp=True):
        
        if disp:
            if not self.loc_ann_subtitles:
                print('Annotated subtitles are not specified!')
            else:
                print('Annotated subtitles are loaded from "{:s}"'.format(self.loc_ann_subtitles))

            if not self.loc_out_subtitles:
                print('Save location is not specified!')
            else:
                print('Save location is "{:s}"'.format(self.loc_out_subtitles))
       
        return self.loc_ann_subtitles, self.loc_out_subtitles
    
    def getFilePaths(self):
        files = os.listdir(self.loc_ann_subtitles)
        return [self.loc_ann_subtitles+f for f in files]
    
    
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
            if disp: print(pattern)
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
                print(match)
                print(m)
                print(s_highlighted)
                print(s+'\n')
                
        #print('\n\n')
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
            #print('cleaned: '+sentences_processed[i])
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
    
    def getAllCleanedProgramSentences(self,disp=False):
        file_paths = self.getFilePaths()

        all_program_id = [None]*len(file_paths)
        all_sentences = [None]*len(file_paths)
        all_sentences_id = [None]*len(file_paths)
        all_highlights = [None]*len(file_paths)
        all_highlights_ind = [None]*len(file_paths)
        
        total_claims = 0
        total_sentences = 0
        
        for f in range(len(file_paths)):
            all_program_id[f], all_sentences_id[f], sentences = \
                        self.getProgramAndSentences(file_paths[f])
            if disp: print('Program id {:s}'.format(all_program_id[f]))
            
            all_sentences[f], all_highlights[f], all_highlights_ind[f] = \
                        self.getCleanedProgramSentences(sentences)
            
            num_claims = len(list(filter(None,all_highlights[f])))
            if disp: print('\tThere were {:d} claims out of {:d} sentences ({:2.2f}%)'.format(num_claims
                    ,len(sentences), num_claims/float(len(sentences))*100))
                
            total_claims = total_claims+num_claims
            total_sentences = total_sentences + len(sentences)
            
        if disp: print('\nIn total there were {:d} claims out of {:d} sentences ({:2.2f}%)'.format(total_claims
                , total_sentences, total_claims/float(total_sentences)*100))
        
        # ...
        labels = ['program_id', 'sentence_id', 'sentence', 'claim_idx', 'claim']
        
        data = [ [None]*len(labels) for i in range(total_sentences)]
        
        
        i = 0
        for p in range(len(all_program_id)):
            
            for si in range(len(all_sentences[p])):
                data[i][0] = all_program_id[p]
                
                data[i][1] = all_sentences_id[p][si]
                data[i][2] = all_sentences[p][si]
                
                if len(all_highlights_ind[p][si]) == 1:
                    data[i][3] = all_highlights_ind[p][si]
                    data[i][4] = all_highlights[p][si]
                    
                elif all_highlights_ind[p][si]:
                    
                    print('HELP')
                    print(all_program_id[p])
                    #print(all_sentences[p][si])
                    print(all_highlights_ind[p][si])
                    print(all_highlights[p][si])
                    new_s, new_idx = self.processMultiClaim(all_sentences[p][si],
                                                      all_highlights_ind[p][si])
                    
                    print('Trying to handle this multi-claim, is the output correct?')
                    print(new_idx)
                    print(new_s)
                    print()
                    
                    data[i][3] = new_idx
                    data[i][4] = new_s
                
                i = i+1
            
        return data, labels