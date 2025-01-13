function [cARSscore] = ChimeraARS(targetSeq, sID, referenceSeqs, suffixArray, alphabet, overlap)

%---------------------------------------------------------------------------------------------------------------------------------               
% Inputs:       string targetSeq: the target sequence
%               -----------------
%
%               double sID: the sequence ID, which is its position in the referenceSeqs cell array if there is an overlap,
%               ----------- if there is no overlap (as with heterologous genes), this parameter is meaningless 
%
%               cell referenceSeqs: cell array of the reference sequences
%               -------------------
%
%               double 3-column array suffixArray
%               ---------------------------------
%               col-1: the sequence ID, which is its position in the referenceSeqs cell array
%               col-2: the start position of the suffix in the sequence
%               col-3: a boolean column indicating if the suffix exists solely in this sequence, 
%                      this is used if the target sequence overlaps the reference sequences
%
%               double alphabet
%               ---------------
%               0 = nucleotides 
%               1 = amino acids
%               2 = codons
%
%               double overlap: indicates if to exclude the target sequence from the reference sequences,
%               --------------- as is necessary for example when dealing with endogenous genes.
%                               The assumption here when using overlap mode is that the reference genome is identical to the target set,
%                               thus removal of each target is performed according to its index. If you wish to run only a subset of the reference
%                               genome as the target set, you need to handle each targets removal from the suffix array defferently.
%               0 = no overlap between target and reference sequences
%               1 = the target sequences overlap the reference ones.
%                  
% Outputs:      double cARSscore: the mean over the maximum subsequence length of each of the targets codon/nucleotide positions, 
%               ----------------- that can be found in at least one of the reference sequences 
%               
% Description:  This function performs the ChimeraARS algorithm: 
%               A measure for the adaptation of the target sequence (targetSeq) to the intracellular gene expression regulatory machinery.
%               
% Copyright:    Tuller Lab, for non commercial use only.
%               zurhadas@post.tau.ac.il; tamirtul@post.tau.ac.il
%               
%---------------------------------------------------------------------------------------------------------------------------------

    global SEQID; global START; global REM; 
    
    % The assumption here when using overlap mode is that the reference genome is identical to the target set,
    % thus removal of each target is performed according to its index. If you wish to run only a subset of the reference
    % genome as the target set, you need to handle each targets removal from the suffix array defferently.
    if (overlap)
        [suffixArray] = remSuffixArray(sID, suffixArray);
    end
    
    if ((alphabet == 0) || (alphabet == 1))
        jump = 1;
    elseif (alphabet == 2)
        jump = 3;
    end
    
    %% Build common suffix array of the target sequence that is found in the reference sequences, and all its prefixes 
    % (note that it is possible to implement ChimeraARS without this step, we do this here to improve the run time)    
    [LCSSubs] = buildCSA(targetSeq, suffixArray, referenceSeqs, jump); 
    
    clear suffixArray;
    
    %% ChimeraARS Main Logic
    
    lcsMAX = max(cellfun(@length, LCSSubs));
    if ((alphabet == 0) || (alphabet == 1))
        lcs = nan(length(targetSeq), 1);
    elseif (alphabet == 2)
        lcs = nan(length(targetSeq)/3, 1);
    end
    
    % For each position i in the target sequence, the longest sub-sequence that starts in that position, 
    % and also appears in at least one of the reference sequences
    count = 1;
    for (i=1:jump:length(targetSeq))
        [lcs(count)] = chimeraARS(i, lcsMAX, LCSSubs, targetSeq, jump);
        count = count+1;
    end
    cARSscore = mean(lcs);
    
end

%%

function [bmap] = chimeraARS(i, lcsMAX, LCSSubs, targetSeq, jump)

    for (j=lcsMAX:-jump:1)
        if (i+j-1 <= length(targetSeq))
            [s, r] = searchSuffix(targetSeq(i:i+j-1), LCSSubs);
            if (~isnan(s) && (s ~= 0) && (s ~= length(LCSSubs)+1))
                bmap = length(i:i+j-1);
                return;
            end
        end
    end
    bmap = 0;
    
end

function [s, r] = searchSuffix(suffix, LCSSubs)

    sortedL = sort({suffix; LCSSubs{1}});
    sortedR = sort({suffix; LCSSubs{end}});
    % 1. suffix is smaller than anything in LCSSubs
    if (find(strcmp(suffix, sortedL)) == 1)
        s = 0;
        r = 0;
    % 2. suffix is larger than anything in LCSSubs
    elseif (find(strcmp(suffix, sortedR)) == 2)
        s = length(LCSSubs)+1;
        r = length(LCSSubs)+1;
    else
        l = 1; r = length(LCSSubs);
        while (l < r)
            mid = floor((l+r)/2);
            sorted = sort({suffix; LCSSubs{mid}});
            if (find(strcmp(suffix, sorted)) == 2)
                l = mid + 1;
            else
                r = mid;
            end
        end
        s = l; r = length(LCSSubs);
        while (l < r)
            mid = ceil((l+r)/2);
            if (strncmp(suffix, LCSSubs{mid}, length(suffix)))
                l = mid;
            else
                r = mid - 1;
            end
        end
    end
    
    if ((s == r) && (s ~= 0) && (s ~= length(LCSSubs)+1))
        if (~strncmp(suffix, LCSSubs{s}, length(suffix)))
            s = NaN;
            r = NaN;
        end
    end
    
end

%% Removes the target sequence suffixes from the suffix array

function [suffixArray] = remSuffixArray(sID, suffixArray)

    global SEQID; global REM;
     
    toRem = find(suffixArray(:,REM) & (suffixArray(:,SEQID) == sID));
    suffixArray(toRem,:) = [];

end

