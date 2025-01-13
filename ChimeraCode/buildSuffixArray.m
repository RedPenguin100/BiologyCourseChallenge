function [suffixArray, allSeqs] = buildSuffixArray(cType, alphabet, refrenceSeqs)

%----------------------------------------------------------------------------------------------               
% Inputs:       double cType: Chimera algorithm type.
%               -------------
%               0 = ChimeraARS
%               1 = ChimeraMap
%
%               double alphabet
%               ---------------
%               0 = nucleotides 
%               1 = amino acids
%               2 = codons
%
%               cell referenceSeqs: cell array of the refrence sequences
%               -------------------
%                  
% Outputs:      if cType = 0:
%               =============
%               double 3-column array suffixArray
%               ---------------------------------
%               col-1: the sequence ID, which is its position in the referenceSeqs cell array
%               col-2: the start position of the suffix in the sequence
%               col-3: a boolean column indicating if the suffix exists solely in this sequence, 
%                      this is used if the target sequences overlap the reference sequences
%
%               if cType = 1:
%               =============
%               double 2-column array suffixArray
%               ---------------------------------
%               col-1: the sequence ID, which is its position in the referenceSeqs cell array
%               col-2: the start position of the suffix in the sequence
%
%               cell allSeqs: a cell array, the same row length as the suffix array, 
%               ------------- of all the sequences containing the suffixes in the suffix array
%               
% Description:  This function builds a suffix array for the Chimera algorithms
%               
% Copyright:    Tuller Lab, for non commercial use only.
%               zurhadas@post.tau.ac.il; tamirtul@post.tau.ac.il
%               
%---------------------------------------------------------------------------------------------- 

    global SEQID; global START; global REM; 

    % Initialize suffix array randomly with the first sequence/gene
    allSeqs = [];
    [posArr] = buildSA(refrenceSeqs{1}, alphabet);
    if (cType == 1)
        suffixArray = [ones(length(posArr),1), posArr];
        allSeqs = cell(length(posArr),2);
        allSeqs(:,SEQID) = {1};
        allSeqs(:,START) = num2cell(posArr);
    else
        suffixArray = [ones(length(posArr),1), posArr, ones(length(posArr),1)];
    end
    
    % Merge the rests of the sequences/genes suffix arrays
    numSeqs = size(refrenceSeqs,1);
    for (i=2:numSeqs)
        [cPosArr] = buildSA(refrenceSeqs{i}, alphabet);
        if (cType == 1)
            cSuffixArray = [zeros(length(cPosArr),1)+i, cPosArr];
        else
            cSuffixArray = [zeros(length(cPosArr),1)+i, cPosArr, ones(length(cPosArr),1)];
        end
        [suffixArray, allSeqs] = merge(suffixArray, cSuffixArray, refrenceSeqs, allSeqs, cType);
    end

end

%%

function [posArr] = buildSA(seq, alphabet)

    if (alphabet == 2)
        jump = 3;
        suffixCell = cell(length(seq)/3,1);
        posArr = nan(length(seq)/3, 1);
    else
        jump = 1;
        suffixCell = cell(length(seq),1);
        posArr = nan(length(seq), 1);
    end
    
    len = length(seq);
    count = 1;
    for (i = 1:jump:len)
        suffixCell{count} = seq(i:len);
        posArr(count) = i;
        count = count+1;
    end
    
    [suffixCell, IX] = sort(suffixCell);
    posArr = posArr(IX);
    
end

%%

function [suffixArray, allSeqs] = merge(suffixArray, cSuffixArray, refrenceSeqs, allSeqs, cType)

    global SEQID; global START; global REM;

    maxLen = size(cSuffixArray,1);
    idxs = nan(maxLen,1);
    is = nan(maxLen,1);
    if (cType == 1)
        cAllSeqs = cell(maxLen,2);
        cAllSeqs(:,SEQID) = {cSuffixArray(1,SEQID)};
        cAllSeqs(:,START) = num2cell(cSuffixArray(:,START));
    end
    count = 1;
    for (i=1:size(cSuffixArray,1))
        curr = refrenceSeqs{cSuffixArray(i, SEQID)}(cSuffixArray(i, START):end);
        [idx, flag] = binarySearch(suffixArray, refrenceSeqs, curr);
        if (flag)
            if (cType == 1)
                allSeqs{idx,SEQID} = [allSeqs{idx,SEQID}, cSuffixArray(i,SEQID)];
                allSeqs{idx,START} = [allSeqs{idx,START}, cSuffixArray(i,START)];
            else
                suffixArray(idx,REM) = 0;
            end
        % new suffix
        else
           idxs(count) = idx+length(find(~isnan(idxs)))+1;
           is(count) = i; 
           count = count+1;
        end
    end
    stay = find(~isnan(idxs));
    idxs = idxs(stay);
    is = is(stay);
    
    if (cType == 1)
        [suffixArray, allSeqs] = updateSuffix(idxs, suffixArray, is, cSuffixArray, allSeqs, cAllSeqs);
    else
        [suffixArray] = updateSuffix(idxs, suffixArray, is, cSuffixArray);
    end
    
end

function [nSuffixArray, nAllGenes] = updateSuffix(fidxs, suffixArray, is, cSuffixArray, allSeqs, cAllSeqs)

    global SEQID; global START; global REM;
    
    % update suffixArray
    nSuffixArray = nan(size(suffixArray,1)+length(fidxs),2);    
    nSuffixArray(fidxs,SEQID) = cSuffixArray(is,SEQID);
    nSuffixArray(fidxs,START) = cSuffixArray(is,START);
    if (nargin == 4)
        nSuffixArray(fidxs,REM) = cSuffixArray(is,REM);
    end
    oidxs = find(isnan(nSuffixArray(:,SEQID)));
    nSuffixArray(oidxs,:) = suffixArray;
    
    % update sequence/gene cell
    if (nargin == 6)
        nAllGenes = cell(size(suffixArray,1)+length(fidxs),2);
        nAllGenes(fidxs,:) = cAllSeqs(is,:);          
        nAllGenes(oidxs,:) = allSeqs;
    else
        nAllGenes = [];
    end

end

function [idx, flag] = binarySearch(suffixArray, refrenceSeqs, str)

    % returns the closest/excat row index of str in sorted table tab
    % returns 0 if str is not found
    
    global SEQID; global START;

    [nrows]=size(suffixArray,1);

    flag = 1;
    lo=1;
    hi=nrows;

    while (lo <= hi)

        idx = fix((lo+hi)/2);
        
        tab = refrenceSeqs{suffixArray(idx, SEQID)}(suffixArray(idx, START):end);
        if (strcmp(str, tab))
            return;
        else
            sorted = sort({str; tab});
            if (find(strcmp(str, sorted)) == 1)
                hi = idx - 1;
            else
                lo = idx + 1;
            end
        end
    end
    
    idx = hi;
    flag = 0;

end

