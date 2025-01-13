function [suffixArray, cARSscores, engineeredTargets, allSeqs] = ChimeraTestExample(cType, alphabet, overlap)

%-----------------------------------------------------------------------------------------------------------------------------------------------               
% Usage:
% ======
% Run ChimeraARS
% --------------
% ChimeraTestExample(0, 0, 0) = run on nucleotide alphabet with no overlap (see below the meaning of overlap)
% ChimeraTestExample(0, 1, 0) = run on amino acid alphabet with no overlap
% ChimeraTestExample(0, 2, 0) = run on codon alphabet with no overlap
% ChimeraTestExample(0, 0, 1) = run on nucleotide alphabet with overlap 
% ChimeraTestExample(0, 1, 1) = run on amino acid alphabet with overlap
% ChimeraTestExample(0, 2, 1) = run on codon alphabet with overlap
% Run ChimeraMap
% --------------
% ChimeraTestExample(1, 1, 0) = run on amino acid alphabet with no overlap
% ChimeraTestExample(1, 1, 1) = run on amino acid alphabet with overlap
%
% General Parameters:
% ===================
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
%               0 = preprocessing: build suffix array
%               1 = run Chimera algorithm
%               
%               cell referenceSeqsNT: cell array of the refrence sequences in nucleotide form
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
% Description:  This function enables running all modes and steps of the Chimera algorithms
%               
% Copyright:    Tuller Lab, for non commercial use only.
%               zurhadas@post.tau.ac.il; tamirtul@post.tau.ac.il
%               
%-----------------------------------------------------------------------------------------------------------------------------------------------

    %% Test Toy Data 
        
    referenceSeqsNT = {'ATGAATGCTGCTATTTTCCGCTTCTTTTTTTACTTTAGCACCTGA';...
                       'ATGAAAGCAATTTTCGTACTGAAAGGTTGGTGGCGCACTTCCTGA';...
                       'ATGAAACACATACCGTTTTTCTTCGCATTCTTTTTTACCTTCCCCTGA';...
                       'ATGACACGCGTTCAATTTAAACACCACCATCATCACCATCATCCTGACTAG';...
                       'ATGATTGAACGTGAACTGGGGAACTGGAAAGACTTTATCGAAGTTATGCTTCGTAAGTAA';...
                       'ATGAAACGCATTAGCACCACCATTACCACCACCATCACCATTACCACAGGTAACGGTGCGGGCTGA';...
                       'ATGAATATCTTACATATATGTGTGACCTCAAAATGGTTCAATATTGACAACAAAATTGTCGATCACCGCCCTTGA';...
                       'ATGGTGGGCCGTTATCGCTTTGAGTTCATTCTGATCATCCTTATTTTATGCGCACTGATTACCGCCCGTTTTTATCTTTCCTGA';...
                       'ATGACTCACATCGTTCGCTTTATCGGTCTACTACTACTAAACGCATCTTCTTTGCGCGGTAGACGAGTGAGCGGCATCCAGCATTAA';...
                       'ATGATACGCATTATCTCAAGAGCAAATTCTGTCACTTCTTCTAATGAAGTGAACCGCTTAGTAACAGGACAGATTCCGCATGACTGA';...
                       'GTGAGTGCAGGCGTGATAACCGGCGTATTGCTGGTGTTTTTATTACTGGGTTATCTGGTTTATGCCCTGATCAATGCGGAGGCGTTCTGA';...
                       'ATGAACCTGGTGGATATCGCCATTCTTATCCTCAAACTCATTGTTGCAGCACTGCAACTGCTTGATGCTGTTCTGAAATACCTGAAGTAA';...
                       'ATGAGAAGCTTCGACCAAGGTTCGACTCGAGCGCCAGCGAGAGAGCGTTGCCGCAGGCAACGACCCGAAGGGCGAAGCGCGCAGCGCTGA';...
                       'ATGAGCACCGACCTTAAATTTTCACTGGTAACAACGATTATCGTCCTCGGTTTGATCGTAGCCGTGGGTTTGACTGCCGCGCTGCACTGA';...
                       'ATGTGGTATTTACTTTGGTTCGTCGGCATTTTGTTGATGTGTTCGCTCTCCACCCTTGTGTTGGTATGGCTGGACCCGCGTCTGAAAAGTTAA';...
                       'ATGAATGTTTCCAGTAGAACTGTAGTACTGATAAATTTCTTTGCTGCTGTTGGTTTGTTTACTCTTATCTCTATGAGATTTGGCTGGTTTATTTGA';...
                       'ATGCTGGGTAATATGAATGTTTTTATGGCCGTACTGGGAATAATTTTATTTTCTGGTTTTCTGGCCGCGTATTTCAGCCACAAATGGGATGACTAA';...
                       'ATGACAGCCCTTCTACGAGTGATTAGCCTGGTCGTGATTAGCGTGGTGGTGATTATTATCCCACCGTGCGGGGCTGCACTTGGACGAGGAAAGGCTTAG';...
                       'ATGACTACTTCCATGCTCAACGCAAAACTACTACCAACTGCGCCATCCGCCGCAGTGGTCGTCGTGCGTGTGGTGGTGGTCGTCGGCAATGCGCCGTAG'};
    
    % The assumption here when using overlap mode is that the reference genome is identical to the target set,
    % thus removal of each target is performed according to its index. If you wish to run only a subset of the reference
    % genome as the target set, you need to handle each targets removal from the suffix array defferently.
    if (overlap)
        targetSeqsNT = referenceSeqsNT;
    else
        targetSeqsNT = {'ATGCGCACATACAATCCAAACTCTCTTCTCCCTTCACAGATGCAGAAATGCACCTGCAATTCTTTGCATCTAGCGTTTGACCTCTGCGGAGGGGAAGCGTGA';...
                        'ATGAACGATCAAATGTTTGTCGAGACACTGATTATCACGTCATCGTTTTTTGCCATTGCTGTTGTACTGGTTTTGTCCGTTCTTCTTATCGAACGGACCGGCTAG';...
                        'ATGACGCTCGCGCAGTTTGCCATGATTTTCTGGCACGACCTGGCAGCACCGATCCTGGCGGGAATTATTACCGCAGCGATTGTCAGCTGGTGGCGTAACCGGAAGTAA';...
                        'ATGACGTTCGCAGAGCTGGGCATGGCCTTCTGGCATGATTTAGCGGCTCCGGTCATTGCTGGCATTCTTGCCAGTATGATCGTGAACTGGCTGAACAAGCGGAAGTAA';...
                        'ATGCCGACTAAACGCTTTGATAAAAAACACTGGAAGATGGTGGTGGTGCTACTGGCAATCTGTGGCGCTATGTTGTTGCTACGTTGGGCAGCAATGATTTGGGGCTGA';...
                        'ATGCGCATAGCTAAAATTGGGGTCATCGCCCTGTTCCTGTTTATGGCGTTAGGCGGAATTGGTGGCGTCATGCTCGCAGGTTATACCTTTATTTTGCGTGCTGGCTAA'};
    end

    
    
    %% ChimeraARS flow
    
    if (cType == 0) % Indicates we are running ChimeraARS
    
        % 1. Build suffix array
        cStep = 0; % Indicates we are building the suffix array
        % The assumption here when using overlap mode is that the reference genome is identical to the target set,
        % thus removal of each target is performed according to its index. If you wish to run only a subset of the reference
        % genome as the target set, you need to handle each targets removal from the suffix array defferently.
        [suffixArray] = Chimera(cType, alphabet, cStep, referenceSeqsNT, targetSeqsNT, overlap);

        % 2. Run ChimeraARS
        cStep = 1; % Indicates we are running the ChimeraARS algorithm utilizing the suffix array we built in the previous stage
        % The output of this step is cARSscore, suffixArray and allSeqs are to be ignored
        [suffixArray, allSeqs, cARSscores] = Chimera(cType, alphabet, cStep, referenceSeqsNT, targetSeqsNT, overlap, suffixArray); 
        engineeredTargets = [];
        suffixArray = suffixArray;
        cARSscores = cARSscores;
        allSeqs = allSeqs;
        disp(allSeqs)
    %% ChimeraMap flow
    
    elseif (cType == 1) % Indicates we are running ChimeraMap

        alphabet = 1; % Override: ChimeraMap is run only on amino acids in this version
        
        % 1. Build suffix array
        cStep = 0; % Indicates we are building the suffix array
        [suffixArray, allSeqs] = Chimera(cType, alphabet, cStep, referenceSeqsNT, targetSeqsNT, overlap);
        
        % 2. Run ChimeraARS
        cStep = 1; % Indicates we are running the ChimeraMap algorithm utilizing the suffix array and allSeqs array we built in the previous stage
        % The output of this step is engineeredTargets; cARSscore, suffixArray and allSeqs are to be ignored
        [suffixArray, allSeqs, cARSscores, engineeredTargets] = Chimera(cType, alphabet, cStep, referenceSeqsNT, targetSeqsNT, overlap, suffixArray, allSeqs);
        suffixArray = suffixArray;
        cARSscores = [];
        engineeredTargets = engineeredTargets;
        allSeqs = allSeqs;
        
    end

end

