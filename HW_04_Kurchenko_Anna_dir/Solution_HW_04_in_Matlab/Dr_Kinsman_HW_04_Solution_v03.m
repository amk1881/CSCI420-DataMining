% Executes all parts of HW_04 and displays any discussed figures
function Dr_Kinsman_HW_04_Solution_v03()


    warning('This code assumes that the attributes are all integer values, and takes advantage of that.');


    % Get all the csvs
    % Give the function a regular expression which matches all possible directories:
    %
    % AND, this does the floor() operation to do noise removal.
    %
    data = Read_all_TRAINING_Data( [ 'TraffStn*' filesep() '*.csv' ] );
    
    % Let's now worry about the ROC curves yet.

    % Initialize to sentiel values:
    best_classifier.which_attribute = -1;	% Sentinel values
    best_classifier.which_threshold = -1;	% Sentinel values
    best_classifier.which_direction = '<='; % Or >, depending on how the nodes come out.
    best_classifier.lowest_mistakes = Inf;	% Sentinel values

    all_intentions = data.INTENT;



    for attribute_test_index = 1 : (size(data,2)-1)
    
        one_attribute           = data(:,attribute_test_index);

        these_attribute_values  = one_attribute.Variables;

	    min_of_attribute   	    = min(these_attribute_values);
	    max_of_attribute   	    = max(these_attribute_values);
        % 
        % Do the noise removal.
        these_attribute_values = floor( these_attribute_values);
    
	    % For each possible threshold speed, from low to high
	    for threshold = min_of_attribute : max_of_attribute                 %#ok<ALIGN>
        
            % Split the data based on this threshold.
            
            % WHAT THIS STEP DOES:
            % This step splits the data, based on this possible threshold.
            % 
            % For each threshold, it forms a CONFUSION MATRIX of how the data
            % would be split, if this threshold were to be used.
            %
            % REMEMBER:
            % A confusion matrix looks like this:
            %
            %                                   ACTUAL VALUES
            %                                   +-----------------+------------+
            %                                   | NON-AGRESSIVE   |  AGRESSIVE |
            %                                   +-----------------+------------+
            %                   NON-AGRESSIVE   |  # Corr. Reject |  # Misses  |
            %     CLASSIFIER    ----------------+-----------------+------------+
            %     GUESS         AGGRESSIVE      |  # False Alarms |  # HITS    |
            %                                   +-----------------+------------+
            %
            %
            %  BUT, HOWEVER, PAY ATTENTION : 
            %  Over the threshold does not mean Agressive!!
            %
            %  Remember, it is possible that the target value is <= the threshold.
            %  you must account for this.
            %
            %  It could be that all agressive drivers dive black cars!! 
            %
            bool_value_leq_threshold = these_attribute_values <= threshold;

            leq_intentions              = all_intentions(  bool_value_leq_threshold );
            gt__intentions              = all_intentions( ~bool_value_leq_threshold );

            n_aggressive_leq_threshold  = sum( leq_intentions == 2 );
            n_behaved_leq_threshold     = sum( leq_intentions ~= 2 );

            n_aggressive_gt_threshold   = sum( gt__intentions == 2 );
            n_behaved_gt_threshold      = sum( gt__intentions ~= 2 );

            % If there are more agressives on the left than the right:
            if ( n_aggressive_leq_threshold > n_aggressive_gt_threshold )
                % Then if <= threshold, you are agressive.
                % Add up the two types of mistakes.
                n_mistakes                  = n_behaved_leq_threshold + n_aggressive_gt_threshold;
                direction_of_classifier     = -1;
            else
                % Then if > threshold, you are agressive.
                % Add up the two mistakes.
                n_mistakes                  = n_behaved_gt_threshold + n_aggressive_leq_threshold;
                direction_of_classifier     = +1;
            end

            if ( n_mistakes < best_classifier.lowest_mistakes )
                best_classifier.which_attribute     = attribute_test_index;
                best_classifier.which_threshold     = threshold;
                best_classifier.direction           = direction_of_classifier;
                best_classifier.lowest_mistakes     = n_mistakes;
            end

        end     % For each threshold of this attribute.
    end         % For each attribute

    Create_Classifier("Dr_Kinsmans_Resulting_Classifier_For_HW04__Go_Create_your_own", ...
            best_classifier.which_attribute, ...
            best_classifier.which_threshold, ...
            best_classifier.direction );

end


%Reads in all the traffic station CSV Files:
function all_data = Read_all_TRAINING_Data( directory_string )

    %Get all the csv files of all the TRAINING data:
    all_csvs = dir(directory_string);

    all_data = [];
    %Read in all data into a single vector
    for index = 1 : numel(all_csvs)

        fn_in = [all_csvs(index).folder  filesep()  all_csvs(index).name];

        directory_data = readtable( fn_in );

        all_data = cat( 1, all_data, directory_data );
    end

    % Truncate to the nearest mile per hour, brightness, etc...
    all_data = floor(all_data);
end



% Creates the one-rule function in a given file for a given threshold
function Create_Classifier(filename,which_attribute,the_threshold, which_direction)

    %opens a file with the given name
    file_pointer = fopen(filename + ".m",'wt');

    fprintf(file_pointer,"%% CAUTION: This code is pseudo-code so far.\n\n");
  
    fprintf(file_pointer,"%% It contains meta-comments to myself.\nn');\n");
    fprintf(file_pointer,"%% It is notes to myself about what to do later on.\n\n");
  

    % create the inital function
    fprintf(file_pointer, "function " + filename + "( filename_in )\n");

    fprintf(file_pointer, "    THE_IMPORTANT_ATTRIBUTE = %d; \n", which_attribute  );
    fprintf(file_pointer, "    THE_IMPORTANT_THRESHOLD = %d; \n", the_threshold    );

    %Get the given csv, even if in a subfolder
    fprintf(file_pointer, "    all_csvs       = readdata( filename_in ), or whatever...\n");
    fprintf(file_pointer, "    the_TEST_data  =  read in all of the data from filename_in ;\n");

    %Read in all data into a single vector  fprintf(file_pointer, 'this code needs to read in all of the data from filename_in\n');

    fprintf(file_pointer, "    %% Truncate the new data to the nearest integer\n");
    fprintf(file_pointer, "    %% Caution: In general, each attribute needs to be noise-cleaned separately.\n");
    fprintf(file_pointer, "    %% Just saying... \n");

    fprintf(file_pointer, "    the_data = floor(data);\n");
    fprintf(file_pointer, "\n");

    if ( which_direction > 0 )
        fprintf(file_pointer, '    %% FOUND THAT AGRESSIVE DRIVERS WERE > THE THRESHOLD for THE_IMPORTANT_THRESHOLD.\n');
        fprintf(file_pointer, "    n_agressive = sum( the_TEST_data(:,THE_IMPORTANT_ATTRIBUTE) > THE_IMPORTANT_THRESHOLD );\n");
        fprintf(file_pointer, "    n_behaving  = sum( the_TEST_data(:,THE_IMPORTANT_ATTRIBUTE) > THE_IMPORTANT_THRESHOLD );\n");
    else
        fprintf(file_pointer, '    %% FOUND THAT AGRESSIVE DRIVERS WERE <= THE THRESHOLD for THE_IMPORTANT_THRESHOLD.\n');
        fprintf(file_pointer, "    n_agressive = sum( the_TEST_data(:,THE_IMPORTANT_ATTRIBUTE) <= THE_IMPORTANT_THRESHOLD );\n");
        fprintf(file_pointer, "    n_behaving  = sum( the_TEST_data(:,THE_IMPORTANT_ATTRIBUTE) <= THE_IMPORTANT_THRESHOLD );\n");
    end
    fprintf(file_pointer, "\n");

    fprintf(file_pointer, "  --- HEY!! Finish this code yourselves  ---  \n");
    fprintf(file_pointer, "  --- I EXPECT YOU TO DO IT IN python, simply because most people do not know Matlab.  :-)  ---  \n");

    % print out the number of cars below the threshold
    fprintf(file_pointer, "    fprintf('n_behaving_well = %%d\\n', n_behaving  );\n");
    fprintf(file_pointer, "    fprintf('n_agressive     = %%d\\n', n_agressive );\n");
    fprintf(file_pointer, "\n");  % You can never have too much white space.

    fprintf(file_pointer, "end    %% End of the classifier function\n");
    fprintf(file_pointer, "\n");

    % closes the input file.
    %
    % GENERAL PRINCIPLE OF COMPUTER SCIENCE:  You do it, you clean it up.
    % If you allocate memory, you need to free it up.
    % If you open a file, you need to close it.
    %
    % YOU is the object oriented "Object" or "Who".
    fclose(file_pointer);
end




