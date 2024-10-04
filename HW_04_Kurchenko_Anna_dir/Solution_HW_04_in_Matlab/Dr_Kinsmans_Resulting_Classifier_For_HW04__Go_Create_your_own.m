% CAUTION: This code is pseudo-code so far.

% It contains meta-comments to myself.
n');
% It is notes to myself about what to do later on.

function Dr_Kinsmans_Resulting_Classifier_For_HW04__Go_Create_your_own( filename_in )
    THE_IMPORTANT_ATTRIBUTE = 1; 
    THE_IMPORTANT_THRESHOLD = 62; 
    all_csvs       = readdata( filename_in ), or whatever...
    the_TEST_data  =  read in all of the data from filename_in ;
    % Truncate the new data to the nearest integer
    % Caution: In general, each attribute needs to be noise-cleaned separately.
    % Just saying... 
    the_data = floor(data);

    % FOUND THAT AGRESSIVE DRIVERS WERE > THE THRESHOLD for THE_IMPORTANT_THRESHOLD.
    n_agressive = sum( the_TEST_data(:,THE_IMPORTANT_ATTRIBUTE) > THE_IMPORTANT_THRESHOLD );
    n_behaving  = sum( the_TEST_data(:,THE_IMPORTANT_ATTRIBUTE) > THE_IMPORTANT_THRESHOLD );

  --- HEY!! Finish this code yourselves  ---  
  --- I EXPECT YOU TO DO IT IN python, simply because most people do not know Matlab.  :-)  ---  
    fprintf('n_behaving_well = %d\n', n_behaving  );
    fprintf('n_agressive     = %d\n', n_agressive );

end    % End of the classifier function

