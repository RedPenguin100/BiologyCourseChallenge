function [suffixArray, allSeqs, cARSscores, engineeredTargets] = Chimera(cType, alphabet, cStep, referenceSeqsNT, targetSeqsNT, overlap, suffixArray, allSeqs)

%-----------------------------------------------------------------------------------------------------------------------------------------------------------           
% Inputs:       double cType: Chimera algorithm type.
%               ------------
%               0 = ChimeraARS
%               1 = ChimeraMap
%               
%               double alphabet
%               ---------------
%               0 = nucleotides 
%               1 = amino acids
%               2 = codons
%               
%               double cStep
%               ------------
%               0 = build suffix array
%               1 = run chimera algorithm
%               
%               cell referenceSeqsNT: cell array of the reference sequences in nucleotide form
%               ---------------------
%
%               cell targetSeqsNT: cell array of the target sequences in nucleotide form
%               ------------------
%
%               double overlap: indicates if to exclude each target gene from the reference genome, 
%               --------------- as is necessary for example when dealing with endogenous genes.
%                               The assumption here when using overlap mode is that the reference genome is identical to the target set,
%                               thus removal of each target is performed according to its index. If you wish to run only a subset of the reference
%                               genome as the target set, you need to handle each targets removal from the suffix array defferently.
%               0 = no overlap between target and reference sequences
%               1 = the target sequences overlap the reference ones.
%
%               if cType = 0 && cStep = 1:
%               ==========================
%               double 3-column array suffixArray
%               ---------------------------------
%               col-1: the sequence ID, which is its position in the referenceSeqs cell array
%               col-2: the start position of the suffix in the sequence
%               col-3: a boolean column indicating if the suffix exists solely in this sequence, 
%                      this is used if the target sequences overlap the reference sequences
%
%               if cType = 1 && cStep = 1:
%               ==========================
%               double 2-column array suffixArray
%               ---------------------------------
%               col-1: the sequence ID, which is its position in the referenceSeqs cell array
%               col-2: the start position of the suffix in the sequence
%
%               cell allSeqs: a cell array, the same row length as the suffix array, 
%               ------------- of all the sequences containing the suffixes in the suffix array
%               
% Outputs:      if cType = 0 && cStep = 0:
%               ==========================
%               double 3-column array suffixArray
%               ---------------------------------
%               col-1: the sequence ID, which is its position in the referenceSeqs cell array
%               col-2: the start position of the suffix in the sequence
%               col-3: a boolean column indicating if the suffix exists solely in this sequence, 
%                      this is used if the target sequences overlap the reference sequences
%
%               if cType = 1 && cStep = 0:
%               ==========================
%               double 2-column array suffixArray
%               ---------------------------------
%               col-1: the sequence ID, which is its position in the referenceSeqs cell array
%               col-2: the start position of the suffix in the sequence
%
%               cell allSeqs: a cell array, the same row length as the suffix array, 
%               ------------- of all the sequences containing the suffixes in the suffix array
%
%               if cType = 0 && cStep = 1:
%               ==========================
%               double array cARSscores: for each target sequence: the mean over the maximum subsequence length of each of its codon/nucleotide positions, 
%               ------------------------ that can be found in at least one of the reference sequences 
%
%               if cType = 1 && cStep = 1:
%               ==========================
%               cell array engineeredTargets: the ChimeraMap engineered version of the target sequences 
%               -----------------------------
%               
% Description:  This function enables running all steps of the Chimera algorithms
%               
% Copyright:    Tuller Lab, for non commercial use only.
%               zurhadas@post.tau.ac.il; tamirtul@post.tau.ac.il
%               
%-----------------------------------------------------------------------------------------------------------------------------------------------------------

    % Suffix Array col indices
    global SEQID; SEQID = 1; global START; START = 2; global REM; REM = 3;

    % Default outputs
    if (cType == 0)
        allSeqs = []; 
    end
    cARSscores = []; engineeredTargets = []; ntSeqs = []; cmapRes = []; imapRes = [];
    
    % Ensure sequences are upper case
    referenceSeqsNT = upper(referenceSeqsNT);
    targetSeqsNT = upper(targetSeqsNT);
    
    % Get amino acid version of targets and reference
    referenceSeqsAA = nt2aa(referenceSeqsNT, 'AlternativeStartCodons', false);
    if (cStep == 1)
        targetSeqsAA = nt2aa(targetSeqsNT, 'AlternativeStartCodons', false);
    end
    
    % Build suffix array
    if (cStep == 0) 
        if (cType == 0) % ChimreaARS build suffix array
            if ((alphabet == 0) || (alphabet == 2))
                [suffixArray] = buildSuffixArray(cType, alphabet, referenceSeqsNT);
            elseif (alphabet == 1)
                [suffixArray] = buildSuffixArray(cType, alphabet, referenceSeqsAA);
            end
        elseif (cType == 1) % ChimeraMap build suffix array
            [suffixArray, allSeqs] = buildSuffixArray(cType, alphabet, referenceSeqsAA);
        end
    % Run Chimera algorithm
    elseif (cStep == 1) 
        % ChimreaARS 
        if (cType == 0) 
            cARSscores = nan(size(targetSeqsNT,1),1); % the mean over the maximum subsequence length of each targets codon/nucleotide positions, 
                                                      % that can be found in at least one of the reference sequences
            for (sID=1:length(targetSeqsNT))
                if ((alphabet == 0) || (alphabet == 2))
                    [cARSscores(sID)] = ChimeraARS(targetSeqsNT{sID}, sID, referenceSeqsNT, suffixArray, alphabet, overlap);
                elseif (alphabet == 1)
                    [cARSscores(sID)] = ChimeraARS(targetSeqsAA{sID}, sID, referenceSeqsAA, suffixArray, alphabet, overlap);
                end
            end
        % ChimeraMap
        elseif (cType == 1)  
            engineeredTargets = cell(size(targetSeqsNT,1),1); %the ChimeraMap engineered version of the target sequences
            for (sID=1:length(targetSeqsNT))
                [engineeredTargets{sID}] = ChimeraMap(targetSeqsAA{sID}, sID, referenceSeqsAA, referenceSeqsNT, suffixArray, overlap, allSeqs);
            end
        end        
    end

end