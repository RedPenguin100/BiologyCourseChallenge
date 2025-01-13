function [engineeredTarget] = ChimeraMap(targetSeq, sID, referenceSeqsAA, referenceSeqsNT, suffixArray, overlap, allSeqs)

%--------------------------------------------------------------------------------------------------------------------------------------------------------               
% Inputs:       string targetSeq: the target sequence
%               -----------------
%
%               double sID: the sequence ID, which is its position in the referenceSeqs cell array if there is an overlap,
%               ----------- if there is no overlap (as with heterologous genes/sequences), this parameter is meaningless 
%
%               cell referenceSeqsAA: cell array of the refrence sequences in amino acid form
%               ---------------------
%
%               cell referenceSeqsNT: cell array of the refrence sequences in nucleotide form
%               ---------------------
%
%               double 2-column array suffixArray
%               ---------------------------------
%               col-1: the sequence ID, which is its position in the referenceSeqs cell array
%               col-2: the start position of the suffix in the sequence
%
%               double overlap: indicates if to exclude the target sequence from the reference sequences,
%               --------------- as is necessary for example when dealing with endogenous mfRefSeqs.
%                               The assumption here when using overlap mode is that the reference genome is identical to the target set,
%                               thus removal of each target is performed according to its index. If you wish to run only a subset of the reference
%                               genome as the target set, you need to handle each targets removal from the suffix array defferently.
%               0 = no overlap between target and reference sequences
%               1 = the target sequences overlap the reference ones.
%
%               cell allSeqs: a cell array, the same row length as the suffix array, 
%               ------------- of all the sequences containing the suffixes in the suffix array
%                  
% Outputs:      string engineeredTarget: the ChimeraMap engineered version of the target sequence 
%               ------------------------
%               
% Description:  This function performs the ChimeraMAP algorithm: 
%               An algorithm for engineering endogenous and heterologous genes/sequences without prior knowledge and based only on the genome of the host
%               
% Copyright:    Tuller Lab, for non commercial use only.
%               zurhadas@post.tau.ac.il; tamirtul@post.tau.ac.il
%               
%--------------------------------------------------------------------------------------------------------------------------------------------------------

    % The assumption here when using overlap mode is that the reference genome is identical to the target set,
    % thus removal of each target is performed according to its index. If you wish to run only a subset of the reference
    % genome as the target set, you need to handle each targets removal from the suffix array defferently.
    if (overlap)
        [suffixArray, allSeqs] = remSuffixArray(sID, suffixArray, allSeqs);
    end
    
    %% Build common suffix array of the target sequence that is found in the reference sequences, and all its prefixes 
    %  (note that it is possible to implement ChimeraMap without this step, we do this here to improve the run time)
    jump = 1;
    [LCSSubs, LCSAllSeqsCell] = buildCSA(targetSeq, suffixArray, referenceSeqsAA, jump, allSeqs);
    clear suffixArray; clear allGenes;
    
    %% ChimeraMap Main Logic
    
    [cmap, imap, smap] = CMap(targetSeq, LCSSubs, LCSAllSeqsCell, referenceSeqsNT);
 
    % construct the engineered nucleotide sequence
    ntSeqs = smap{end}; % the covering subsequences induced by the ChimeraMap blocks
    cmapRes = cmap{end}; % ChimeraMap blocks
    imapRes = imap{end}; % % the reference sequence id/s from which the blocks were derived
    % We are not outputting the above 3 parameters for simplicity, but those who want to utilize them in their research can.
    engineeredTarget = [];
    for (i=1:length(ntSeqs))
        engineeredTarget = [engineeredTarget ntSeqs{i}];
    end
    

end

%%

function [cmap, imap, smap] = CMap(targetSeq, LCSSubs, LCSAllSeqsCell, referenceSeqsNT)

    % For debugging purposes we keep the entire flow of the algorithm, this can be written only maintaining the previous optimal solution at each step
    cmap = cell(length(targetSeq), 1); % ChimeraMap blocks
    imap = cell(length(targetSeq), 1); % the reference sequence id/s from which the blocks were derived
    smap = cell(length(targetSeq), 1); % the covering subsequences induced by the ChimeraMap blocks
    for (i=1:length(targetSeq))
        if (i>1)
            [pmap, pimap, psmap] = chimeraMap(i, cmap, imap, smap, LCSSubs, targetSeq, LCSAllSeqsCell, referenceSeqsNT);
        else
            % We assume here that all AAs are found in the reference sequences
            % The code needs to be modified if this assumption does not hold
            inds = find(strncmp(targetSeq(1:i), LCSSubs, 1));
            if (~isempty(inds))
                pmap = [1, i];
                [best, psmap, pimap] = findMostFreq(inds, LCSAllSeqsCell, referenceSeqsNT, targetSeq(1:i));
            end
        end
        
        cmap{i} = pmap;
        if (i == 1)
            smap{i} = {psmap};
            imap{i} = {pimap};
        else
            smap{i} = psmap;
            imap{i} = pimap;
        end
    end
       
end

%%

function [pmap, pimap, psmap] = chimeraMap(i, cmap, imap, smap, LCSSubs, targetSeq, LCSAllSeqsCell, referenceSeqsNT)

    pmap = cmap{i-1};
    pimap = imap{i-1};
    psmap = smap{i-1};

    [best1, seq, mfRefSeqs] = findMostFreq(find(strncmp(targetSeq(pmap(end,1):i), LCSSubs, length(targetSeq(pmap(end,1):i)))), LCSAllSeqsCell, referenceSeqsNT, targetSeq(pmap(end,1):i));

    if (best1)
        pmap(end,2) = i;
        pimap{end,1} = mfRefSeqs;
        psmap{end,1} = seq;
    else
        [best2, seq, mfRefSeqs] = findMostFreq(find(strncmp(targetSeq(pmap(end,2)+1:i), LCSSubs, length(targetSeq(pmap(end,2)+1:i)))), LCSAllSeqsCell, referenceSeqsNT, targetSeq(pmap(end,2)+1:i));
        if (best2) 
            e2 = pmap(end,2);
            pmap(end+1,:) = [e2+1, i];
            pimap{end+1,1} = mfRefSeqs;
            psmap{end+1,1} = seq;
        end
    end

end

%%

function [best, seq, mfRefSeqs] = findMostFreq(inds, LCSAllSeqsCell, referenceSeqsNT, target)

    global SEQID; global START;
    
    if (isempty(inds))
        best = 0; % indicates failure
        seq = []; % the covering subsequence selected
        mfRefSeqs = []; % ids of the reference sequences containing the covering subsequence selected 
        return;
    end
    
    seqMap = java.util.HashMap; % potential covering subsequences map
    idMap = java.util.HashMap; % ids of the reference sequences containing the potential covering subsequences
    for (i=1:length(inds))
        mfRefSeqs = LCSAllSeqsCell{inds(i), SEQID};
        starts = LCSAllSeqsCell{inds(i), START};
        for (j=1:length(mfRefSeqs))
            seq = referenceSeqsNT{mfRefSeqs(j)}(starts(j)*3-2:(starts(j)*3-2)+length(target)*3-1);
            if (isempty(seqMap.get(seq)))
                seqMap.put(seq, 1);
                idMap.put(seq, mfRefSeqs(j));
            else
                seqMap.put(seq, seqMap.get(seq)+1);
                arr = idMap.get(seq);
                arr(end+1,1) = mfRefSeqs(j);
                idMap.put(seq, arr);
            end
        end
    end
    freqs = [seqMap.keySet.toArray.cell, seqMap.values.toArray.cell];
    freqs = sortrows(freqs, 1);
    [maxi, I] = max(cell2mat(freqs(:, 2)));
    seqs = seqMap.keySet.toArray.cell;
    seq = seqs{I}; % the covering subsequence selected
    mfRefSeqs = unique(idMap.get(seq)); % ids of the reference sequences containing the covering subsequence selected 
    best = 1; % indicates success
    
end

%%

function [suffixArray, allSeqs] = remSuffixArray(sID, suffixArray, allSeqs)

    global SEQID; 
    
    toRem = [];
    for (i=1:length(allSeqs(:,SEQID)))
        if ((length(allSeqs{i,SEQID}) == 1) && (allSeqs{i,SEQID} == sID))
            toRem(end+1,1) = i; 
        end
    end
    allSeqs(toRem,:) = [];
    suffixArray(toRem,:) = [];

end


