function [LCSSubs, LCSAllSeqsCell] = buildCSA(targetSeq, suffixArray, referenceSeqs, jump, allSeqs)

%-------------------------------------------------------------------------------------------------------------------------------------------------------               
% Inputs:       string targetSeq: the target sequence
%               -----------------
%
%               double 3-column array suffixArray
%               ---------------------------------
%               col-1: the sequence ID, which is its position in the referenceSeqs cell array
%               col-2: the start position of the suffix in the sequence
%               col-3: a boolean column indicating if the suffix exists solely in this sequence, 
%                      this is used if the target sequence overlaps the reference sequences
%
%               cell referenceSeqs: cell array of the refrence sequences
%               -------------------
%
%               double jump: indicates the alphabet used
%               ------------
%               jump = 1: amino acids or nucleotides
%               jump = 3: codons
%
%               cell allSeqs: a cell array, the same row length as the suffix array, 
%               ------------- of all the sequences containing the suffixes in the suffix array, relevant only for ChimeraMap
%                  
% Outputs:      if ChimeraARS:
%               ==============
%               cell array LCSSubs: the common suffix array of the target sequence that is found in the reference sequences, and all its prefixes
%               -------------------
%
%               if ChimeraMap:
%               ==============
%               cell array LCSSubs: the common suffix array of the target sequence that is found in the reference sequences, and all its prefixes
%               -------------------
%               cell array LCSAllSeqsCell:
%               --------------------------
%               col-1: the sequence ID, which is its position in the referenceSeqs cell array, of all the sequences containing the suffix
%               col-2: the start position of the suffix in all the sequences containing the suffix
%            
%               
% Description:  This function builds the common suffix array of the target sequence that is found in the reference sequences, and all its prefixes
%               
% Copyright:    Tuller Lab, for non commercial use only.
%               zurhadas@post.tau.ac.il; tamirtul@post.tau.ac.il
%               
%-------------------------------------------------------------------------------------------------------------------------------------------------------

    LCSSubs = [];
    LCSAllSeqsCell = {};
    
    count = 1;
    for (si=1:jump:length(targetSeq))
        if (nargin == 5) % indicates ChimeraMap mode
            [LCSSubs, count, LCSAllSeqsCell] = findCSA(targetSeq(si:end), suffixArray, referenceSeqs, LCSSubs, count, jump, allSeqs, LCSAllSeqsCell);
        else
            [LCSSubs, count] = findCSA(targetSeq(si:end), suffixArray, referenceSeqs, LCSSubs, count, jump);
        end
    end
    
    % Do not check if substring already exists as computationally takes more time, perform unique at the end of the process
    [LCSSubs, IX] = unique(LCSSubs);
    if (nargin == 5)
        LCSAllSeqsCell = LCSAllSeqsCell(IX, :);
        [LCSAllSeqsCell] = makeUnique(LCSAllSeqsCell);
    end

end

%%

function [LCSSubs, count, LCSAllSeqsCell] = findCSA(suffix, suffixArray, referenceSeqs, LCSSubs, count, jump, allSeqs, LCSAllSeqsCell)

    global SEQID; global START;
    
    if (nargin == 6)
        LCSAllSeqsCell = [];
    end
    
    for (i=jump:jump:length(suffix))

        [s, r] = searchSuffix(suffix(1:i), suffixArray, referenceSeqs);
        if (~isnan(s) && (s ~= 0) && (s ~= length(suffixArray)+1))
            if (s ~= r)
                arr = [s:r];
            else
                arr = s;
            end

            % Do not check if substring already exists as computationally takes more time, perform unique at the end of the process
            LCSSubs{count,1} = suffix(1:i);
            %
            if (nargin == 8)
                LCSAllSeqsCell{count, SEQID} = allSeqs(arr,SEQID);
                LCSAllSeqsCell{count, START} = allSeqs(arr,START);
            end
            %
            count = count+1;

        else
            break;
        end
        
    end
    
end


%%

function [s, r] = searchSuffix(suffix, suffixArray, referenceSeqs)

    global SEQID; global START;

    sortedL = sort({suffix; referenceSeqs{suffixArray(1,SEQID)}(suffixArray(1,START):end)});
    sortedR = sort({suffix; referenceSeqs{suffixArray(length(suffixArray),SEQID)}(suffixArray(length(suffixArray),START):end)});
    % 1. suffix is smaller than anything in suffixArray
    if (~strncmp(referenceSeqs{suffixArray(1,SEQID)}(suffixArray(1,START):end), suffix, length(suffix)) && (find(strcmp(suffix, sortedL)) == 1))        
        s = 0;
        r = 0;
    % 2. suffix is larger than anything in suffixArray
    elseif (find(strcmp(suffix, sortedR)) == 2)
        s = length(suffixArray)+1;
        r = length(suffixArray)+1;
    else
        l = 1; r = length(suffixArray);
        while (l < r)
            mid = floor((l+r)/2);
            sorted = sort({suffix; referenceSeqs{suffixArray(mid,SEQID)}(suffixArray(mid,START):end)});
            if (find(strcmp(suffix, sorted)) == 2)
                l = mid + 1;
            else
                r = mid;
            end
        end
        s = l; r = length(suffixArray);
        while (l < r)
            mid = ceil((l+r)/2);
            if (strncmp(suffix, referenceSeqs{suffixArray(mid,SEQID)}(suffixArray(mid,START):end), length(suffix)))
                l = mid;
            else
                r = mid - 1;
            end
        end
    end
    
    if ((s == r) && (s ~= 0) && (s ~= length(suffixArray)+1))
        if (~strncmp(suffix, referenceSeqs{suffixArray(s,SEQID)}(suffixArray(s,START):end), length(suffix)))
            s = NaN;
            r = NaN;
        end
    end
    
end

%%

function [LCSAllSeqsCell] = makeUnique(LCSAllSeqsCell)

    global SEQID; global START;
    
    for (i=1:size(LCSAllSeqsCell, 1))
        seqCell = LCSAllSeqsCell{i,SEQID};
        startCell = LCSAllSeqsCell{i,START};
        newSeqs = [];
        newStarts = [];
        for (j=1:length(seqCell))
            newSeqs = [newSeqs seqCell{j}];
            newStarts = [newStarts startCell{j}];
        end
        [newSeqs, IX] = unique(newSeqs);
        newStarts = newStarts(IX);
        LCSAllSeqsCell{i,SEQID} = newSeqs;
        LCSAllSeqsCell{i,START} = newStarts;
    end
end


